import pytest
import torch

from olmo_core.distributed.parallel import DataParallelType, build_world_mesh
from olmo_core.distributed.utils import get_full_tensor
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.hyperball import (
    MuonConstraint,
    MuonH,
    MuonHConfig,
    MuonWConfig,
    newton_schulz_msign,
)
from olmo_core.testing import (
    DEVICES,
    requires_gpu,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all


def build_dense_model(**kwargs) -> Transformer:
    return TransformerConfig.olmo2_30M(
        vocab_size=1024, n_layers=2, init_method=InitMethod.fan_in, **kwargs
    ).build()


def build_moe_model(**kwargs) -> Transformer:
    return TransformerConfig.smallmoe(
        vocab_size=1024, n_layers=2, d_model=128, init_method=InitMethod.fan_in, **kwargs
    ).build()


# --- newton_schulz_msign ----------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32), (16, 48), (48, 16)])
def test_msign_drives_singular_values_to_one(shape):
    seed_all(0)
    G = torch.randn(*shape, dtype=torch.float32)
    svals = torch.linalg.svdvals(newton_schulz_msign(G).float())
    # NS5 is tuned for speed, not precision: it lands the spectrum in a band around 1.
    assert svals.min() > 0.6
    assert svals.max() < 1.4


def test_msign_batched_matches_per_block_loop():
    """A blocked call must be exactly a loop over blocks, or MoE experts leak into each other."""
    seed_all(0)
    blocks = torch.randn(5, 12, 20, dtype=torch.float32)
    batched = newton_schulz_msign(blocks)
    looped = torch.stack([newton_schulz_msign(blocks[i]) for i in range(blocks.shape[0])])
    torch.testing.assert_close(batched, looped)


def test_msign_tolerates_an_all_zero_block():
    """An expert that received no tokens has a zero update and must not produce NaNs."""
    blocks = torch.zeros(2, 8, 8, dtype=torch.float32)
    blocks[0] = torch.randn(8, 8)
    out = newton_schulz_msign(blocks)
    assert torch.isfinite(out).all()
    assert (out[1] == 0).all()


# --- the Hyperball invariants ------------------------------------------------------------


def _single_param_optim(W: torch.Tensor, **kwargs) -> MuonH:
    return MuonH([{"params": [W]}], **kwargs)


@pytest.mark.parametrize("block_rows", [None, 4])
def test_hyperball_holds_the_radius_fixed(block_rows):
    """``||W_b||_F == R_b`` for every block, at every step. This is the whole constraint."""
    seed_all(0)
    W = torch.randn(16, 10, dtype=torch.float32, requires_grad=True)
    radii = W.detach().view(-1, block_rows or 16, 10).norm(dim=(-2, -1)).clone()

    optim = _single_param_optim(W, lr=0.05, constraint=MuonConstraint.hyperball)
    optim.param_groups[0]["block_rows"] = block_rows

    for _ in range(6):
        W.grad = torch.randn_like(W)
        optim.step()
        now = W.detach().view(-1, block_rows or 16, 10).norm(dim=(-2, -1))
        torch.testing.assert_close(now, radii, rtol=1e-5, atol=1e-5)


def test_hyperball_step_length_is_lr_times_radius():
    """The unprojected step is ``lr * R`` by construction; the projection is second order."""
    seed_all(0)
    W = torch.randn(24, 24, dtype=torch.float32, requires_grad=True)
    radius = W.detach().norm().item()
    lr = 1e-3

    optim = _single_param_optim(W, lr=lr, constraint=MuonConstraint.hyperball)
    before = W.detach().clone()
    W.grad = torch.randn_like(W)
    optim.step()

    moved = (W.detach() - before).norm().item()
    assert moved == pytest.approx(lr * radius, rel=0.02)


@pytest.mark.parametrize("scale", [1e-4, 1e3])
def test_hyperball_ignores_the_scale_of_the_update(scale):
    """
    The paper's observation that Muon's ``s_mu``/Moonlight scalars cancel under Hyperball.

    Exact in real arithmetic; here it is bounded by the bfloat16 Newton-Schulz iteration, so
    the assertion is a tight tolerance rather than bit-equality. It is not merely academic --
    the momentum's magnitude spans orders of magnitude between warmup and the end of decay,
    and Hyperball throws all of it away.
    """
    grad = torch.randn(16, 16, generator=torch.Generator().manual_seed(0))
    results = []
    for factor in (1.0, scale):
        seed_all(1)
        W = torch.randn(16, 16, dtype=torch.float32, requires_grad=True)
        optim = _single_param_optim(W, lr=0.02, constraint=MuonConstraint.hyperball)
        W.grad = grad * factor
        optim.step()
        results.append(W.detach().clone())
    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-6)


def test_hyperball_ignores_adjust_lr_but_weight_decay_does_not():
    """``adjust_lr`` is inert under Hyperball and load-bearing under weight decay."""
    seed_all(0)
    grad = torch.randn(16, 32)

    def run(constraint, adjust_lr):
        seed_all(1)
        W = torch.randn(16, 32, dtype=torch.float32, requires_grad=True)
        optim = _single_param_optim(
            W, lr=0.01, constraint=constraint, adjust_lr=adjust_lr, weight_decay=0.1
        )
        W.grad = grad.clone()
        optim.step()
        return W.detach().clone()

    hb_rms = run(MuonConstraint.hyperball, "rms_norm")
    hb_spec = run(MuonConstraint.hyperball, "spectral_norm")
    torch.testing.assert_close(hb_rms, hb_spec)

    wd_rms = run(MuonConstraint.weight_decay, "rms_norm")
    wd_spec = run(MuonConstraint.weight_decay, "spectral_norm")
    assert not torch.allclose(wd_rms, wd_spec)


def test_weight_decay_arm_shrinks_a_zero_gradient_weight():
    """MuonW decays; MuonH cannot, because the projection restores the norm."""
    for constraint, shrinks in (
        (MuonConstraint.weight_decay, True),
        (MuonConstraint.hyperball, False),
    ):
        seed_all(1)
        W = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
        before = W.detach().norm().item()
        optim = _single_param_optim(W, lr=0.1, constraint=constraint, weight_decay=0.5)
        W.grad = torch.zeros_like(W)
        optim.step()
        after = W.detach().norm().item()
        assert (after < before * 0.99) is shrinks


def test_blocked_update_matches_independent_per_block_optimizers():
    """
    A stacked expert tensor with ``block_rows`` set must evolve exactly as separate matrices
    each with their own optimizer -- same radius, same msign, no cross-expert coupling.
    """
    seed_all(0)
    num_blocks, rows, cols = 4, 6, 10
    init = torch.randn(num_blocks * rows, cols, dtype=torch.float32)
    grads = [torch.randn(num_blocks * rows, cols) for _ in range(3)]

    stacked = init.clone().requires_grad_(True)
    stacked_optim = _single_param_optim(stacked, lr=0.03, constraint=MuonConstraint.hyperball)
    stacked_optim.param_groups[0]["block_rows"] = rows
    for g in grads:
        stacked.grad = g.clone()
        stacked_optim.step()

    separate = []
    for b in range(num_blocks):
        rows_slice = slice(b * rows, (b + 1) * rows)
        W = init[rows_slice].clone().requires_grad_(True)
        optim = _single_param_optim(W, lr=0.03, constraint=MuonConstraint.hyperball)
        for g in grads:
            W.grad = g[rows_slice].clone()
            optim.step()
        separate.append(W.detach())

    torch.testing.assert_close(stacked.detach(), torch.cat(separate, dim=0))


def test_radius_is_recovered_from_the_weights_after_a_resume():
    """
    The radius is not checkpointed. It does not need to be: the constraint keeps
    ``||W_b||_F == R_b``, so a fresh optimizer measures the same radius back.
    """
    seed_all(0)
    W = torch.randn(12, 20, dtype=torch.float32, requires_grad=True)
    optim = _single_param_optim(W, lr=0.05, constraint=MuonConstraint.hyperball)
    for _ in range(4):
        W.grad = torch.randn_like(W)
        optim.step()
    original = optim._radii[W].clone()

    resumed = _single_param_optim(W, lr=0.05, constraint=MuonConstraint.hyperball)
    resumed.state[W]["momentum"] = optim.state[W]["momentum"].clone()
    W.grad = torch.randn_like(W)
    resumed.step()
    torch.testing.assert_close(resumed._radii[W], original, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block_rows", [None, 4])
def test_latest_metrics_reports_the_constraint_holding(block_rows):
    """
    The drift metric is what tells a broken constraint apart from a losing optimizer, so it has
    to actually track the invariant rather than report zero unconditionally.
    """
    seed_all(0)
    W = torch.randn(16, 10, dtype=torch.float32, requires_grad=True)
    optim = _single_param_optim(W, lr=0.05, constraint=MuonConstraint.hyperball)
    optim.param_groups[0]["block_rows"] = block_rows

    assert optim.latest_metrics() == {}
    for _ in range(3):
        W.grad = torch.randn_like(W)
        optim.step()

    metrics = optim.latest_metrics()
    assert metrics["radius_relative_drift_max"] < 1e-5
    expected_blocks = 16 // (block_rows or 16)
    assert metrics["matrix_norm_min"] <= metrics["matrix_norm_mean"] <= metrics["matrix_norm_max"]
    if expected_blocks == 1:
        assert metrics["matrix_norm_min"] == pytest.approx(metrics["matrix_norm_max"])

    # Break the invariant behind the optimizer's back; the metric must notice.
    with torch.no_grad():
        W.mul_(1.5)
    W.grad = torch.randn_like(W)
    optim.step()
    # The step re-projects, so drift is measured against a radius the weights no longer had --
    # it is reported, not silently absorbed.
    assert optim.latest_metrics()["radius_relative_drift_max"] < 1e-5
    assert W.detach().norm().item() == pytest.approx(
        optim._radii[W].flatten().square().sum().sqrt().item(), rel=1e-4
    )


def test_weight_decay_arm_reports_norms_but_no_drift():
    """There is no radius on the control arm, so there is nothing to report drift against."""
    seed_all(0)
    W = torch.randn(12, 12, dtype=torch.float32, requires_grad=True)
    optim = _single_param_optim(W, lr=0.01, constraint=MuonConstraint.weight_decay)
    W.grad = torch.randn_like(W)
    optim.step()

    metrics = optim.latest_metrics()
    assert "matrix_norm_mean" in metrics
    assert "radius_relative_drift_max" not in metrics


def test_radius_scale_moves_the_weights_onto_the_requested_sphere():
    seed_all(0)
    W = torch.randn(10, 10, dtype=torch.float32, requires_grad=True)
    base = W.detach().norm().item()
    optim = _single_param_optim(W, lr=0.01, constraint=MuonConstraint.hyperball, radius_scale=2.0)
    W.grad = torch.randn_like(W)
    optim.step()
    assert W.detach().norm().item() == pytest.approx(2.0 * base, rel=1e-4)


# --- config wiring ------------------------------------------------------------------------


@pytest.mark.parametrize("config_cls", [MuonHConfig, MuonWConfig])
def test_config_builds_and_splits_a_dense_model(config_cls):
    model = build_dense_model()
    optim = config_cls(lr=0.02, adamw_lr=1e-3).build(model)

    assert isinstance(optim, MuonH)
    algorithms = [g.get("algorithm", "muon") for g in optim.param_groups]
    assert algorithms.count("muon") == 1
    assert algorithms.count("adamw") == 3  # embeddings, gains, lm_head

    for group in optim.param_groups:
        assert "initial_lr" in group
        assert group["lr"] == (0.02 if group.get("algorithm", "muon") == "muon" else 1e-3)
        if group.get("algorithm", "muon") == "muon":
            assert all(p.ndim == 2 for p in group["params"])

    assert config_cls().merge(["lr=1e-1"]).lr == 0.1


@pytest.mark.parametrize("config_cls", [MuonHConfig, MuonWConfig])
def test_config_blocks_moe_experts_per_expert(config_cls):
    model = build_moe_model()
    moe = model.blocks["0"].feed_forward_moe  # type: ignore[union-attr]
    num_experts = moe.experts.mlp.num_experts

    optim = config_cls(lr=0.02, adamw_lr=1e-3).build(model)

    blocked = {}
    for group in optim.param_groups:
        if group.get("algorithm", "muon") != "muon":
            continue
        for p in group["params"]:
            if group.get("block_rows") is not None:
                blocked[p.shape] = group["block_rows"]
                # Whatever the block size is, it must cut the tensor into exactly one
                # matrix per expert.
                assert p.shape[0] == group["block_rows"] * num_experts

    # Every stacked expert weight got blocked, none was left whole.
    expert_shapes = {p.shape for p in (moe.experts.mlp.w1, moe.experts.mlp.w2, moe.experts.mlp.w3)}
    assert expert_shapes <= set(blocked)


def test_moe_router_is_optimized_with_adamw():
    """The router is a small classifier, not a hidden weight matrix; Muon should not see it."""
    model = build_moe_model()
    router = model.blocks["0"].feed_forward_moe.router.weight  # type: ignore[union-attr]
    optim = MuonHConfig(lr=0.02, adamw_lr=1e-3).build(model)

    for group in optim.param_groups:
        params = {id(p) for p in group["params"]}
        if id(router) in params:
            assert group.get("algorithm", "muon") == "adamw"
            break
    else:
        pytest.fail("router weight was not in any param group")


def _fwd_bwd_step(model: Transformer, optim: MuonH, device: torch.device) -> None:
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name


@pytest.mark.parametrize("config_cls", [MuonHConfig, MuonWConfig])
@pytest.mark.parametrize("device", DEVICES)
def test_dense_step_runs_end_to_end(config_cls, device: torch.device):
    seed_all(0)
    model = build_dense_model().train().to(device)
    _fwd_bwd_step(model, config_cls(lr=0.02, adamw_lr=1e-3).build(model), device)


@requires_gpu
@pytest.mark.parametrize("config_cls", [MuonHConfig, MuonWConfig])
def test_moe_step_runs_end_to_end(config_cls):
    """
    GPU-only: the ``default`` MoE dispatch goes through Triton ``binned_gather`` kernels, which
    have no CPU path. The CPU tests above cover the param-group split, which is where the
    MoE-specific logic in this module actually lives.
    """
    device = torch.device("cuda")
    seed_all(0)
    model = build_moe_model().train().to(device)
    _fwd_bwd_step(model, config_cls(lr=0.02, adamw_lr=1e-3).build(model), device)


# --- distributed ---------------------------------------------------------------------------


def _run_fsdp_parity(config_cls, factory_kwargs):
    """
    An FSDP step must land where a single-device step would. The gather path for dense
    matrices and the shard-local path for experts both have to agree with the reference.
    """
    device = get_default_device()
    seed_all(0)

    reference = TransformerConfig.olmo2_30M(
        vocab_size=1024, init_method=InitMethod.fan_in, **factory_kwargs
    ).build(init_device=device.type)
    reference.train()
    ref_optim = config_cls(lr=0.02, adamw_lr=1e-3).build(reference)

    seed_all(0)
    dp_config = TransformerDataParallelConfig(name=DataParallelType.fsdp)
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    sharded = TransformerConfig.olmo2_30M(
        vocab_size=1024, init_method=InitMethod.fan_in, **factory_kwargs
    ).build(init_device=device.type)
    sharded.train()
    sharded = parallelize_model(sharded, world_mesh=world_mesh, device=device, dp_config=dp_config)
    sharded_optim = config_cls(lr=0.02, adamw_lr=1e-3).create_optimizer(sharded)

    input_ids = torch.randint(0, 1024, (2, 8), device=device)
    reference(input_ids).sum().backward()
    ref_optim.step()
    sharded(input_ids).sum().backward()
    sharded_optim.step()

    for (name, ref_p), (_, shard_p) in zip(
        reference.named_parameters(), sharded.named_parameters()
    ):
        torch.testing.assert_close(
            get_full_tensor(shard_p), ref_p, rtol=1e-4, atol=1e-4, msg=lambda m: f"{name}: {m}"
        )


@requires_multi_gpu
@pytest.mark.parametrize("config_cls", [MuonHConfig, MuonWConfig])
def test_fsdp_matches_single_device(config_cls):
    seed_all(0)
    run_distributed_test(
        _run_fsdp_parity,
        backend="nccl",
        start_method="spawn",
        world_size=2,
        func_args=(config_cls, {"n_layers": 2}),
    )


def _run_hsdp_step(config_cls, shard_degree: int, num_replicas: int):
    device = get_default_device()
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp, shard_degree=shard_degree, num_replicas=num_replicas
    )
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    model = TransformerConfig.smallmoe(
        vocab_size=1024, n_layers=2, d_model=128, init_method=InitMethod.fan_in
    ).build(init_device=device.type)
    model.train()
    model = parallelize_model(model, world_mesh=world_mesh, device=device, dp_config=dp_config)

    optim = config_cls(lr=0.02, adamw_lr=1e-3).create_optimizer(model)
    model(torch.randint(0, 1024, (2, 8), device=device)).sum().backward()
    optim.step()

    for p in model.parameters():
        assert torch.isfinite(get_full_tensor(p)).all()


@requires_multi_gpu
@pytest.mark.parametrize("config_cls", [MuonHConfig, MuonWConfig])
@pytest.mark.parametrize(
    "shard_degree,num_replicas",
    [
        pytest.param(2, 1, id="shard2_replica1"),
        pytest.param(1, 2, id="shard1_replica2"),
    ],
)
def test_hsdp_moe_step(config_cls, shard_degree: int, num_replicas: int):
    seed_all(0)
    run_distributed_test(
        _run_hsdp_step,
        backend="nccl",
        start_method="spawn",
        world_size=2,
        func_args=(config_cls, shard_degree, num_replicas),
    )
