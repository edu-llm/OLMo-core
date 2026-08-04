from __future__ import annotations

import ast
import contextlib
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

EDULLM_ROOT = Path(__file__).resolve().parents[1]
if str(EDULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDULLM_ROOT))

import token_selection_370m.selection as selection_module  # noqa: E402
from token_selection_370m.arms import (  # noqa: E402
    ARM_SPECS,
    REFHQ,
    REFHQ_LATE_STEPS,
)
from token_selection_370m.blade import (  # noqa: E402
    BLADE_CHECKPOINT_FORMAT,
    BLADE_REFERENCE_MICROBATCH_TOKENS,
    BLADE_SELECTION_MICROBATCH_TOKENS,
    BLADE_SYNC_STEPS,
    BladeCallback,
    BladeSchedule,
    ResumableBatchStream,
    _full_proxy_state,
)
from token_selection_370m.recipe import (  # noqa: E402
    ALPHA_F,
    CUSTOM_LOSS_METHODS,
    GLOBAL_BATCH_TOKENS,
    PEAK_LR,
    PRODUCTION_WORLD_SIZE,
    RANK_MICROBATCH_TOKENS,
    SEQUENCE_LENGTH,
    WARMUP_STEPS,
    Z_LOSS,
    immutable_corpus_binding,
    scientific_identity,
    total_steps,
)
from token_selection_370m.precomputed import (  # noqa: E402
    MASK_ALGORITHM,
    binding_sha256,
    load_mask_manifest,
    weights_to_label_mask,
)
from token_selection_370m.selection import (  # noqa: E402
    EMAHistory,
    WeightShadow,
    attention_received_from_qk,
    capture_last_attention,
    ema_alpha,
    selection_weights,
)


def test_exact_approved_arm_family_and_wandb_routing() -> None:
    assert tuple(
        (name, spec.method, spec.dataset_id, spec.keep_fraction) for name, spec in ARM_SPECS.items()
    ) == (
        ("rho-1", "rho_excess", "pretrain/regmix-10b", 0.6),
        ("rel-ema-exp", "rel_ema", "pretrain/regmix-10b", 0.6),
        ("middle-ppl-token", "middle_ppl", "pretrain/regmix-10b", 0.6),
        ("attention", "attention_topk", "pretrain/regmix-10b", 0.6),
        ("blade", "blade", "pretrain/regmix-10b", 0.6),
    )
    assert all(
        ARM_SPECS[name].wandb_project == "token-selection"
        for name in ("rho-1", "rel-ema-exp", "middle-ppl-token", "attention", "blade")
    )
    assert ARM_SPECS["middle-ppl-token"].late_reference_contract.endswith(str(REFHQ_LATE_STEPS))


def test_one_recipe_constants_and_2360_step_budget() -> None:
    assert SEQUENCE_LENGTH == 2048
    assert GLOBAL_BATCH_TOKENS == 4_194_304
    assert RANK_MICROBATCH_TOKENS == 32_768
    assert PEAK_LR == 4e-4
    assert WARMUP_STEPS == 24
    assert ALPHA_F == 0.1
    assert Z_LOSS == 1e-5
    assert total_steps(9_900_000_000) == 2360


def test_custom_module_backpropagates_differentiable_total_loss() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert 'token_loss = self._loss_tensor(output, micro_labels, "loss")' in source
    assert "loss = (token_loss.float() * weights).sum() / divisor" in source


def test_weight_swaps_reshard_fsdp_before_restoring_parameters() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "selection.py").read_text(encoding="utf-8")
    assert source.count("_reshard(model)") == 2
    assert "owner.unshard()" in source
    assert "owner.reshard()" in source


def test_weight_shadow_matches_reference_outputs_and_rho_masks() -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.0, -2.0]))
    reference = Tiny()
    reference_state = {"weight": torch.tensor([3.0, 5.0], dtype=torch.float64)}
    reference.load_state_dict(reference_state)
    original = training.weight.detach().clone()
    inputs = torch.tensor([[2.0, -1.0]])
    targets = torch.tensor([[4.0, -2.0]])

    expected_reference_output = reference(inputs)
    expected_reference_loss = (expected_reference_output - targets).square()
    current_loss = (training(inputs) - targets).square()
    expected_mask = selection_weights(
        "rho_excess",
        valid=torch.ones_like(current_loss, dtype=torch.bool),
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current_loss,
        reference=expected_reference_loss,
    )

    shadow = WeightShadow.from_state_dict(training, reference_state)
    assert torch.equal(training.weight, original)
    assert shadow.weights["weight"].shape == training.weight.shape
    assert shadow.weights["weight"].device == training.weight.device
    assert shadow.weights["weight"].dtype == training.weight.dtype
    with torch.no_grad(), shadow.swap_to(training):
        actual_reference_output = training(inputs)
    actual_reference_loss = (actual_reference_output - targets).square()
    actual_mask = selection_weights(
        "rho_excess",
        valid=torch.ones_like(current_loss, dtype=torch.bool),
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current_loss,
        reference=actual_reference_loss,
    )

    assert torch.equal(actual_reference_output, expected_reference_output)
    assert torch.equal(actual_reference_loss, expected_reference_loss)
    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(training.weight, original)


def test_weight_shadow_repeatedly_restores_training_parameters_and_gradients() -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.25, -2.5]))
    original = training.weight.detach().clone()
    shadow = WeightShadow.from_state_dict(
        training,
        {"weight": torch.tensor([3.0, 5.0])},
    )

    for _ in range(5):
        with torch.no_grad(), shadow.swap_to(training):
            assert torch.equal(training.weight, torch.tensor([3.0, 5.0]))
        assert torch.equal(training.weight, original)

    with pytest.raises(RuntimeError, match="scoring failed"):
        with shadow.swap_to(training):
            raise RuntimeError("scoring failed")
    assert torch.equal(training.weight, original)

    training(torch.tensor([[2.0, -4.0]])).sum().backward()
    assert torch.equal(training.weight.grad, torch.tensor([2.0, -4.0]))
    assert all(
        not weight.requires_grad and weight.grad is None for weight in shadow.weights.values()
    )
    assert torch.equal(training.weight, original)


def test_weight_shadow_hot_path_uses_only_local_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.0, 2.0]))
    writes = 0
    original_write = selection_module._write

    def count_write(parameter, value):
        nonlocal writes
        writes += 1
        original_write(parameter, value)

    monkeypatch.setattr(selection_module, "_write", count_write)
    shadow = WeightShadow.from_state_dict(
        training,
        {"weight": torch.tensor([3.0, 5.0])},
    )
    assert writes == 1

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot path attempted CPU or full-tensor materialization")

    monkeypatch.setattr(selection_module, "_write", forbidden)
    monkeypatch.setattr(selection_module, "_snapshot", forbidden)
    reference_storage = shadow.weights["weight"].data_ptr()
    for _ in range(3):
        with shadow.swap_to(training):
            assert torch.equal(training.weight, torch.tensor([3.0, 5.0]))
        assert torch.equal(training.weight, torch.tensor([1.0, 2.0]))
        assert shadow.weights["weight"].data_ptr() == reference_storage


def test_reference_microbatches_share_one_weight_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = Tiny()
    training.weight.data.copy_(torch.tensor([1.0, 2.0]))
    shadow = WeightShadow.from_state_dict(
        training,
        {"weight": torch.tensor([3.0, 5.0])},
    )
    original_swap = shadow.swap_to
    swaps = 0

    @contextlib.contextmanager
    def counted_swap(model):
        nonlocal swaps
        swaps += 1
        with original_swap(model) as swapped:
            yield swapped

    monkeypatch.setattr(shadow, "swap_to", counted_swap)
    inputs = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[4.0, 7.0]]),
        torch.tensor([[3.0, 6.0]]),
    ]
    with torch.no_grad(), shadow.swap_to(training):
        scores = [training(value) for value in inputs]

    assert swaps == 1
    assert torch.equal(scores[0], torch.tensor([[3.0, 10.0]]))
    assert torch.equal(scores[1], torch.tensor([[12.0, 35.0]]))
    assert torch.equal(scores[2], torch.tensor([[9.0, 30.0]]))
    assert torch.equal(training.weight, torch.tensor([1.0, 2.0]))
    assert training.training


def test_train_module_batches_reference_scoring_structurally() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    score_many = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_score_many"
    )
    swaps = [
        node
        for node in ast.walk(score_many)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "swap_to"
    ]
    assert len(swaps) == 1
    assert any(
        isinstance(child, ast.For)
        for with_node in ast.walk(score_many)
        if isinstance(with_node, ast.With)
        for statement in with_node.body
        for child in ast.walk(statement)
    )
    assert "self._score_many(state.reference, scoring_batches)" in source
    assert "self._score(state.reference" not in source


def test_weight_accounting_synchronizes_once_per_batch() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "train_module.py").read_text(encoding="utf-8")
    assert "observed_weight += weights.sum()" in source
    assert source.count("observed_weight.item()") == 1


def test_precomputed_weights_align_with_shifted_label_masks() -> None:
    weights = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    label_mask = weights_to_label_mask(weights)

    assert torch.equal(label_mask[:, 1:], weights[:, :-1].bool())
    assert not label_mask[:, 0].any()
    assert torch.equal(label_mask[:, 1:].sum(-1), weights[:, :-1].sum(-1))


def test_precomputed_manifest_binds_reference_corpus_and_mask_sizes(tmp_path: Path) -> None:
    sources = [tmp_path / "tokens-0.bin", tmp_path / "tokens-1.bin"]
    masks = [tmp_path / "mask-0.bin", tmp_path / "mask-1.bin"]
    for source in sources:
        source.write_bytes(bytes(range(16)))
    for mask in masks:
        mask.write_bytes(bytes(8))
    source_ids = ["s3://sealed/tokens-0.bin", "s3://sealed/tokens-1.bin"]
    binding = {
        "algorithm": MASK_ALGORITHM,
        "sequence_length": 4,
        "keep_fraction": 0.6,
        "reference_sha256": "abc123",
        "source_ids": source_ids,
        "total_instances": 4,
        "selected_tokens": 8,
        "mask_files": [
            {
                "source_id": source_id,
                "mask_size": 8,
                "mask_sha256": "fixture",
            }
            for source_id in source_ids
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "binding": binding,
                "binding_sha256": binding_sha256(binding),
                "files": [
                    {
                        "source_id": source_id,
                        "source_size": source.stat().st_size,
                        "mask_path": str(mask),
                        "mask_size": mask.stat().st_size,
                        "mask_sha256": "fixture",
                    }
                    for source_id, source, mask in zip(source_ids, sources, masks)
                ],
            }
        ),
        encoding="utf-8",
    )

    paths, actual_binding = load_mask_manifest(
        manifest,
        corpus_paths=[str(source) for source in sources],
        source_ids=source_ids,
        source_itemsize=2,
        sequence_length=4,
        keep_fraction=0.6,
        reference_sha256="abc123",
    )
    assert paths == [str(mask) for mask in masks]
    assert actual_binding == binding

    masks[0].write_bytes(bytes(7))
    with pytest.raises(RuntimeError, match="missing or has changed"):
        load_mask_manifest(
            manifest,
            corpus_paths=[str(source) for source in sources],
            source_ids=source_ids,
            source_itemsize=2,
            sequence_length=4,
            keep_fraction=0.6,
            reference_sha256="abc123",
        )


def test_middle_ppl_precomputed_fast_path_uses_standard_trainer() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "recipe.py").read_text(encoding="utf-8")
    assert "label_mask_paths=list(label_mask_paths)" in source
    assert "arm.method in CUSTOM_LOSS_METHODS and not precomputed_middle_ppl" in source
    assert '"precomputed_selection": precomputed_middle_ppl' in source
    assert 'late_reference_path=None if arm.method == "middle_ppl"' in source


def test_weight_shadow_restores_local_fsdp_parameter_after_forward(tmp_path: Path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized():
        pytest.skip("test requires ownership of the local process group")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "fsdp-store").as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        training = nn.Linear(2, 2, bias=False)
        training.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        reference_state = {"weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]])}
        baseline = nn.Linear(2, 2, bias=False)
        baseline.load_state_dict(reference_state)
        inputs = torch.tensor([[2.0, -1.0]])
        expected = baseline(inputs)

        fully_shard(training, mesh=init_device_mesh("cpu", (1,)))
        original_shard = training.weight.to_local().detach().clone()
        shadow = WeightShadow.from_state_dict(training, reference_state)
        with torch.no_grad(), shadow.swap_to(training):
            actual = training(inputs)

        assert torch.equal(actual, expected)
        assert torch.equal(training.weight.to_local(), original_shard)
        assert shadow.weights["weight"].device == training.weight.to_local().device
        assert shadow.weights["weight"].shape == training.weight.to_local().shape
    finally:
        dist.destroy_process_group()


def test_attention_capture_hooks_compiled_block_boundary() -> None:
    class Attention(nn.Module):
        def forward(self, x, **_kwargs):
            return x

    class CompiledLikeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = Attention()

        def forward(self, x, **kwargs):
            # Simulate a compiled graph that bypasses child-module __call__ hooks.
            return self.attention.forward(x, **kwargs)

    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": CompiledLikeBlock()})
    expected = torch.randn(2, 3, 4)
    with capture_last_attention(model) as capture:
        model.blocks["0"](expected, start_pos=0)
    assert torch.equal(capture.x, expected)
    assert capture.owner is model.blocks["0"]
    assert capture.kwargs == {"start_pos": 0}


def test_method_polarities_and_per_sequence_selection() -> None:
    valid = torch.ones(2, 4, dtype=torch.bool)
    current = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    reference = torch.ones_like(current)
    rho = selection_weights(
        "rho_excess",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current,
        reference=reference,
    )
    assert rho.bool().tolist() == [
        [True, True, False, False],
        [False, False, True, True],
    ]
    rel = selection_weights(
        "rel_ema",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current,
        history=reference * 3,
    )
    assert rel.sum(dim=1).tolist() == [2, 2]
    learn = selection_weights(
        "learnability",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
        early=torch.tensor([[4.0, 3.0, 2.0, 1.0]]).expand(2, -1),
        late=torch.tensor([[1.0, 1.5, 1.5, 3.0]]).expand(2, -1),
    )
    assert learn[0].bool().tolist() == [True, True, False, False]
    blade = selection_weights(
        "blade",
        valid=valid,
        keep_fraction=0.5,
        step=500,
        seed=42,
        current=current,
        reference=reference * 3,
    )
    assert blade.bool().tolist() == [
        [True, True, False, False],
        [False, False, True, True],
    ]


def test_middle_ppl_drops_easy_and_hard_and_random_is_resumable() -> None:
    valid = torch.ones(1, 10, dtype=torch.bool)
    middle = selection_weights(
        "middle_ppl",
        valid=valid,
        keep_fraction=0.6,
        step=0,
        seed=42,
        reference=torch.arange(10.0).unsqueeze(0),
    )
    assert middle.bool().tolist() == [
        [False, False, True, True, True, True, True, True, False, False]
    ]
    first = selection_weights("random", valid=valid, keep_fraction=0.6, step=125, seed=42)
    resumed = selection_weights("random", valid=valid, keep_fraction=0.6, step=125, seed=42)
    assert torch.equal(first, resumed)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


def test_relative_ema_variants_and_resume_state() -> None:
    assert ema_alpha(0, tau=300, constant=None) == 0.0
    assert ema_alpha(300, tau=300, constant=None) == pytest.approx(1 - math.exp(-1))
    assert ema_alpha(100, tau=None, constant=0.9985) == 0.9985

    model = Tiny()
    ema = EMAHistory(model)
    model.weight.data.fill_(2)
    ema.update(model, 0.5)
    model.weight.data.fill_(4)
    ema.update(model, 0.5)
    state = ema.state_dict()
    restored = EMAHistory(Tiny())
    restored.load_state_dict(state)
    assert restored.correction == ema.correction
    assert torch.equal(restored.shadow["weight"], ema.shadow["weight"])

    seeded = EMAHistory(Tiny(), seed={"weight": torch.tensor([3.0, 5.0])})
    assert seeded.correction == 1.0
    with seeded.swap_to(model):
        assert torch.equal(model.weight, torch.tensor([3.0, 5.0]))
    assert torch.equal(model.weight, torch.tensor([4.0, 4.0]))


def test_attention_received_matches_causal_definition() -> None:
    query = torch.zeros(1, 3, 1, 2)
    key = torch.zeros_like(query)
    # Uniform causal attention: received mass is 1 + 1/2 + 1/3, 1/2 + 1/3, 1/3.
    scores = attention_received_from_qk(query, key)
    assert scores[0].tolist() == pytest.approx([11 / 6, 5 / 6, 1 / 3])


class FakeStream:
    def __init__(self, cursor: int):
        self.cursor = cursor

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


def _blade_callback(train_cursor=3, hq_cursor=7) -> BladeCallback:
    return BladeCallback(
        total_steps=2360,
        reference_factory=Tiny,
        reference_train_stream=FakeStream(train_cursor),  # type: ignore[arg-type]
        refhq_stream=FakeStream(hq_cursor),  # type: ignore[arg-type]
    )


def test_blade_proxy_sync_materializes_full_state_on_every_rank(monkeypatch) -> None:
    import torch.distributed.checkpoint.state_dict as dist_cp_state

    captured = {}

    def get_model_state_dict(model, *, options):
        captured["options"] = options
        return model.state_dict()

    monkeypatch.setattr(dist_cp_state, "get_model_state_dict", get_model_state_dict)
    state = _full_proxy_state(Tiny())

    assert set(state) == {"weight"}
    assert captured["options"].full_state_dict is True
    assert captured["options"].cpu_offload is False


def test_blade_k_update_microbatches_with_full_batch_gradient(monkeypatch) -> None:
    callback = _blade_callback()
    callback._new_reference()
    callback.reference_microbatch_tokens = 4
    calls = []
    batch = {"input_ids": torch.arange(8, dtype=torch.long).reshape(4, 2)}

    def mean_ce(model, micro_batch, *, loss_div_factor=None):
        calls.append(micro_batch["input_ids"].clone())
        assert loss_div_factor is not None
        return model.weight.sum() * micro_batch["input_ids"].float().sum() / loss_div_factor

    monkeypatch.setattr(callback, "_mean_ce", mean_ce)
    assert callback.reference is not None
    callback._backward_mean_ce(callback.reference, batch, weight=0.6)

    assert len(calls) == 2
    assert all(call.shape == (2, 2) for call in calls)
    expected = torch.full_like(callback.reference.weight, 0.6 * batch["input_ids"].sum() / 4)
    assert torch.allclose(callback.reference.weight.grad, expected)


def test_blade_selection_scoring_microbatches_full_rank_batch(monkeypatch) -> None:
    callback = _blade_callback()
    callback.selection_microbatch_tokens = 4
    batch = {"input_ids": torch.arange(8, dtype=torch.long).reshape(4, 2)}
    calls = []

    def score_microbatch(micro_batch):
        ids = micro_batch["input_ids"]
        calls.append(ids.clone())
        labels = ids.clone()
        return labels, ids.float() + 1, ids.float() + 3

    monkeypatch.setattr(callback, "_proxy_and_reference_ce_microbatch", score_microbatch)
    labels, proxy_ce, reference_ce = callback._proxy_and_reference_ce(batch)

    assert len(calls) == 2
    assert all(call.shape == (2, 2) for call in calls)
    assert torch.equal(labels, batch["input_ids"])
    assert torch.equal(proxy_ce, batch["input_ids"].float() + 1)
    assert torch.equal(reference_ce, batch["input_ids"].float() + 3)


def test_blade_pre_step_selects_largest_proxy_minus_reference_gap(monkeypatch) -> None:
    callback = _blade_callback()
    callback._new_reference()
    callback.last_sync = 500
    callback.trainer = types.SimpleNamespace(
        global_step=500,
        train_module=types.SimpleNamespace(label_ignore_index=-100),
    )
    labels = torch.tensor([[10, 11, 12, 13]])
    proxy_ce = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    reference_ce = torch.full_like(proxy_ce, 3.0)
    monkeypatch.setattr(
        callback,
        "_proxy_and_reference_ce",
        lambda batch: (labels, proxy_ce, reference_ce),
    )
    batch = {"input_ids": labels.clone()}

    callback.pre_step(batch)

    assert batch["labels"].tolist() == [[10, 11, 12, -100]]


def test_blade_sync_boundary_saves_before_and_after_k_updates(monkeypatch) -> None:
    callback = _blade_callback()
    callback.trainer = types.SimpleNamespace(global_step=499)
    events = []

    def sync():
        events.append("sync")
        if callback.reference is None:
            callback._new_reference()

    monkeypatch.setattr(callback, "_sync_from_proxy", sync)
    monkeypatch.setattr(
        callback,
        "_save_sync_checkpoint",
        lambda *, step, phase: events.append(f"save-{phase}-{step}"),
    )
    monkeypatch.setattr(
        callback,
        "_run_k_updates",
        lambda *, trainer_step=None: events.append(f"k-{trainer_step}"),
    )

    callback.post_train_batch()

    assert events == ["sync", "save-pre-500", "sync", "k-500", "save-post-500"]
    assert callback.completed_step == 499
    assert callback.last_sync == 500


def test_resume_prefers_post_sync_boundary_until_normal_step_catches_up(tmp_path) -> None:
    from token_selection_entrypoint import _latest_resume_checkpoint

    def materialize(path: Path) -> None:
        (path / "model_and_optim").mkdir(parents=True)
        (path / "model_and_optim" / ".metadata").touch()

    normal = tmp_path / "step375"
    pre = tmp_path / "sync_checkpoints" / "step500-pre"
    post = tmp_path / "sync_checkpoints" / "step500-post"
    for path in (normal, pre, post):
        materialize(path)

    assert _latest_resume_checkpoint(tmp_path) == (post, True)

    caught_up = tmp_path / "step500"
    materialize(caught_up)
    assert _latest_resume_checkpoint(tmp_path) == (caught_up, False)


def test_blade_locked_schedule_and_full_resume_state() -> None:
    assert BLADE_SYNC_STEPS == (500, 875, 1250, 1625, 2000)
    assert BLADE_REFERENCE_MICROBATCH_TOKENS == 8_192
    assert BLADE_SELECTION_MICROBATCH_TOKENS == 32_768
    callback = _blade_callback()
    callback._new_reference()
    assert callback.reference is not None and callback.reference_optim is not None
    callback.reference.weight.data.fill_(9)
    callback.completed_step = 1250
    callback.last_sync = 1250
    state = callback.state_dict()
    assert state["checkpoint_format"] == BLADE_CHECKPOINT_FORMAT
    assert state["dynamic_reference_optim"] is not None
    assert state["reference_train_stream"] == {"cursor": 3}
    assert state["refhq_stream"] == {"cursor": 7}

    restored = _blade_callback(0, 0)
    restored._restore(state)
    assert restored.completed_step == 1250
    assert restored.last_sync == 1250
    assert restored.reference is not None
    assert torch.equal(restored.reference.weight, torch.tensor([9.0, 9.0]))
    assert restored.reference_train_stream.cursor == 3
    assert restored.refhq_stream.cursor == 7
    assert next(step for step in BLADE_SYNC_STEPS if step > restored.completed_step) == 1625

    boundary = _blade_callback()
    boundary._new_reference()
    boundary.completed_step = 499
    boundary.last_sync = 500
    restored_boundary = _blade_callback(0, 0)
    restored_boundary._restore(boundary.state_dict())
    assert restored_boundary.completed_step == 499
    assert restored_boundary.last_sync == 500


def test_blade_rejects_schedule_drift_and_missing_post_warmup_reference() -> None:
    with pytest.raises(ValueError, match="locked"):
        BladeSchedule(k_steps=74).validate(2360)
    state = _blade_callback().state_dict()
    state["completed_step"] = 500
    state["last_sync"] = 500
    with pytest.raises(ValueError, match="missing"):
        _blade_callback()._restore(state)
    source = _blade_callback()
    source._new_reference()
    inconsistent = source.state_dict()
    inconsistent["completed_step"] = 1250
    inconsistent["last_sync"] = 875
    with pytest.raises(ValueError, match="last sync"):
        _blade_callback()._restore(inconsistent)


class FakeResumableLoader:
    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor
        self.epoch = 0

    def reshuffle(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return self

    def __next__(self):
        value = self.cursor
        self.cursor += 1
        return {"cursor": value}

    def state_dict(self):
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]


def test_blade_secondary_stream_resume_is_next_batch_exact() -> None:
    original = ResumableBatchStream(FakeResumableLoader())
    assert original.next() == {"cursor": 0}
    state = original.state_dict()
    expected = original.next()
    resumed = ResumableBatchStream(FakeResumableLoader())
    resumed.load_state_dict(state)
    assert resumed.next() == expected


def test_identity_pins_reference_provenance_and_fixture(tmp_path: Path) -> None:
    arm = ARM_SPECS["rho-1"]
    reference = tmp_path / "refhq-step1315.pt"
    reference.write_bytes(b"immutable reference")
    corpus = types.SimpleNamespace(
        version="v1",
        paths=("s3://edullm-data/pretrain/regmix-10b/v1/train-00000.bin",),
        dtype="<u4",
        rows=9_900_000_000,
    )
    binding = immutable_corpus_binding(arm.dataset_id, corpus)
    identity = scientific_identity(
        arm,
        dataset_binding=binding,
        refhq_binding=None,
        max_tokens=9_900_000_000,
        reference_path=str(reference),
        early_reference_path=None,
        late_reference_path=None,
    )
    assert identity["reference_contract"] == arm.reference_contract
    assert len(identity["reference_sha256"]) == 64
    assert identity["dataset_binding"] == binding
    assert len(identity["dataset_binding"]["paths_sha256"]) == 64
    assert identity["wandb_project"] == "token-selection"
    fixture = json.loads(
        (EDULLM_ROOT / "platform" / "token-selection-arms.json").read_text(encoding="utf-8")
    )
    assert set(fixture["arms"]) == set(ARM_SPECS)
    assert fixture["arms"]["blade"]["secondary_dataset_release"] == REFHQ


def test_immutable_bindings_fail_closed_for_latest_and_missing_blade_refhq() -> None:
    unresolved = types.SimpleNamespace(
        version="latest",
        paths=("s3://example/train.bin",),
        dtype="<u4",
        rows=1,
    )
    with pytest.raises(ValueError, match="immutable version"):
        immutable_corpus_binding("pretrain/regmix-10b", unresolved)

    resolved = types.SimpleNamespace(
        version="v1",
        paths=("s3://example/train.bin",),
        dtype="<u4",
        rows=1,
    )
    with pytest.raises(ValueError, match="RefHQ binding"):
        scientific_identity(
            ARM_SPECS["blade"],
            dataset_binding=immutable_corpus_binding("pretrain/regmix-10b", resolved),
            refhq_binding=None,
            max_tokens=9_900_000_000,
            reference_path=None,
            early_reference_path=None,
            late_reference_path=None,
        )


def test_production_recipe_statically_assembles_public_olmo_apis() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "recipe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert {
        "TransformerConfig.olmo2_370M",
        "NumpyFSLDatasetConfig",
        "NumpyDataLoaderConfig",
        "TrainerConfig",
        "CheckpointerCallback",
        "GPUMemoryMonitorCallback",
        "ConfigSaverCallback",
        "WandBCallback",
        "TaskLossEvalCallback",
        "trainer_config.build",
    } <= calls
    assert "DataParallelType.hsdp" in source
    assert "LoadStrategy.if_available if resume else LoadStrategy.never" in source
    assert 'checkpoint_kwargs["pre_train_checkpoint"] = not resume' in source
    assert "module_config.build(model)" in source
    assert "_custom_module(model, module_config, selection_config)" in source
    assert 'checkpoint_kwargs["fixed_steps"]' in source
    assert "task_loss_nproc=PRODUCTION_WORLD_SIZE if production else None" in source
    assert PRODUCTION_WORLD_SIZE == 8
    assert {spec.method for spec in ARM_SPECS.values()} - CUSTOM_LOSS_METHODS == {"blade"}
    assert CUSTOM_LOSS_METHODS >= {
        "rho_excess",
        "rel_ema",
        "middle_ppl",
        "attention_topk",
    }


def test_platform_entrypoint_is_locked_to_eight_gpu_torchrun() -> None:
    launcher = (EDULLM_ROOT / "platform" / "entrypoint.sh").read_text(encoding="utf-8")
    fixture = json.loads(
        (EDULLM_ROOT / "platform" / "token-selection-arms.json").read_text(encoding="utf-8")
    )
    assert "python -m torch.distributed.run" in launcher
    assert "--nproc_per_node=8" in launcher
    assert fixture["compute_profile"] == "gpu-8xa100"
    assert fixture["gpu_count"] == 8
    assert fixture["entrypoint"] == "bash .edullm/platform/entrypoint.sh"
    submission = json.loads(
        (EDULLM_ROOT / "fixtures" / "token-selection-attention-submission.json").read_text(
            encoding="utf-8"
        )
    )
    command = submission["command"][-1]
    assert submission["compute_profile"] == "gpu-8xa100"
    assert submission["workload_profile"] == "olmo-core-train-4gpu"
    assert submission["dataset_release"] == "regmix-10b-v1"
    assert submission["wandb_project"] == "token-selection-attention"
    assert "--nproc-per-node=8" in command
    assert "EDULLM_CHECKPOINT_CHECK=waived" in command

    benchmark = json.loads(
        (
            EDULLM_ROOT / "fixtures" / "token-selection-attention-benchmark-submission.json"
        ).read_text(encoding="utf-8")
    )
    benchmark_command = benchmark["command"][-1]
    assert benchmark["compute_profile"] == "gpu-8xa100"
    assert benchmark["maximum_attempts"] == 1
    assert benchmark["wandb_project"] == "token-selection-attention"
    assert "--nproc-per-node=8" in benchmark_command
    assert "--arm attention" in benchmark_command
    assert "--local" in benchmark_command
    assert "WANDB_MODE=disabled" in benchmark_command
    assert "WANDB_API_KEY" not in benchmark_command
    assert "/opt/olmo-core/.edullm/eval_task_loss_olmo_core.py" in benchmark_command

    handoff = (EDULLM_ROOT / "README-token-selection.md").read_text(encoding="utf-8")
    assert "1.25 * T" in handoff
    assert "run-approval-admin" in handoff
    assert "submit in table-index order" in handoff
    assert (
        str(EDULLM_ROOT / "eval_task_loss_olmo_core.py")
        .replace("\\", "/")
        .endswith(".edullm/eval_task_loss_olmo_core.py")
    )


def test_packaged_evaluator_and_image_are_complete() -> None:
    evaluator = EDULLM_ROOT / "eval_task_loss_olmo_core.py"
    tree = ast.parse(evaluator.read_text(encoding="utf-8"))
    labels = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_LABELS" for target in node.targets
        ):
            labels = ast.literal_eval(node.value)
            break
    from production_contract.task_loss import TASK_LOSS_RAW_LABELS

    assert labels == TASK_LOSS_RAW_LABELS
    docker = (EDULLM_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-token-selection-eval.txt" in docker
    assert "py_compile .edullm/eval_task_loss_olmo_core.py" in docker
    requirements = (EDULLM_ROOT / "requirements-token-selection-eval.txt").read_text(
        encoding="utf-8"
    )
    assert "090253dac6688f2532509daa7aa2eb5fae50e956" in requirements
    assert "transformers==4.57.6" in requirements


def test_entrypoint_builds_and_fits_production_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import token_selection_entrypoint as entrypoint

    task_script = tmp_path / "task-loss.py"
    task_script.write_text("# fixture\n", encoding="utf-8")
    events: list[str] = []

    class Corpus:
        version = "immutable-v1"
        rows = 9_900_000_000
        paths = ("s3://edullm-data/pretrain/regmix-10b/v1/train-00000.bin",)
        dtype = "<u4"

    class Trainer:
        def fit(self) -> None:
            events.append("fit")

    def fake_build(*args, **kwargs):
        assert args[0] is ARM_SPECS["attention"]
        assert kwargs["production"] is True
        assert kwargs["resume"] is False
        events.append("build")
        return Trainer()

    fake_olmo = types.ModuleType("olmo_core")
    fake_train = types.ModuleType("olmo_core.train")
    fake_utils = types.ModuleType("olmo_core.utils")
    fake_train.prepare_training_environment = lambda **kwargs: events.append("prepare")
    fake_train.teardown_training_environment = lambda: events.append("teardown")
    fake_utils.seed_all = lambda seed: events.append(f"seed:{seed}")
    monkeypatch.setitem(sys.modules, "olmo_core", fake_olmo)
    monkeypatch.setitem(sys.modules, "olmo_core.train", fake_train)
    monkeypatch.setitem(sys.modules, "olmo_core.utils", fake_utils)
    monkeypatch.setattr(entrypoint, "resolve_corpus", lambda **kwargs: Corpus())
    monkeypatch.setattr(entrypoint, "build_trainer", fake_build)
    monkeypatch.setattr(entrypoint, "write_identity", lambda *args: events.append("identity"))
    monkeypatch.setattr(entrypoint, "assert_production_runtime", lambda: events.append("world:8"))
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "token_selection_entrypoint.py",
            "--arm",
            "attention",
            "--save-folder",
            str(tmp_path / "save"),
            "--work-dir",
            str(tmp_path / "work"),
            "--progress-dir",
            str(tmp_path / "progress"),
            "--task-loss-script",
            str(task_script),
        ],
    )

    entrypoint.main()

    assert events == [
        "prepare",
        "world:8",
        "seed:6198",
        "build",
        "identity",
        "fit",
        "teardown",
    ]
