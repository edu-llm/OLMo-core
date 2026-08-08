from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest
import torch

from olmo_core.nn.residual_stream import HyperConnectionConfig
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.callbacks import HyperConnectionMonitorCallback

VOCAB_SIZE = 128
D_MODEL = 64
SEQ_LEN = 16
WEIGHT_DECAY = 0.033


def build_model(hc: HyperConnectionConfig):
    config = TransformerConfig.llama_like(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=2,
        n_heads=4,
        block_name=TransformerBlockType.hyper_connection_reordered_norm,
        qk_norm=True,
    )
    assert not isinstance(config.block, dict)
    config.block.hyper_connections = hc
    model = config.build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=SEQ_LEN)
    return model


@dataclass
class FakeTrainModule:
    model: Any


@dataclass
class FakeTrainer:
    """
    Stands in for the trainer, and reproduces the two rules of ``Trainer.record_metric`` that
    a callback can get wrong: a metric recorded twice in one step needs a merge strategy, and
    a metric's reduce type has to be the same every time it is recorded.
    """

    train_module: FakeTrainModule
    global_step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    reduce_types: Dict[str, Any] = field(default_factory=dict)
    seen_this_step: set = field(default_factory=set)

    def record_metric(self, name: str, value, reduce_type=None, merge_strategy=None, **kwargs):
        del kwargs
        key = (self.global_step, name)
        if key in self.seen_this_step and merge_strategy is None:
            raise AssertionError(
                f"'{name}' recorded twice at step {self.global_step} with no merge strategy; "
                "the real trainer warns and keeps the first value"
            )
        if name in self.reduce_types and self.reduce_types[name] != reduce_type:
            raise AssertionError(
                f"'{name}' changed reduce type from {self.reduce_types[name]} to {reduce_type}; "
                "the real trainer raises"
            )
        self.reduce_types[name] = reduce_type
        self.seen_this_step.add(key)
        self.metrics[name] = float(value)


def attach(model, **kwargs) -> HyperConnectionMonitorCallback:
    callback = HyperConnectionMonitorCallback(interval=1, **kwargs)
    callback.trainer = FakeTrainer(FakeTrainModule(model))  # type: ignore[assignment]
    callback.post_attach()
    callback.pre_train()
    return callback


def measured_blocks(callback, spreads: List[float]):
    """
    Seed the monitor with one reading per block, as if a model of that depth had just run a
    training step, and let the guard read them back through its own code path.
    """
    callback._lane_spreads = {f"blocks.{i}": v for i, v in enumerate(spreads)}
    callback._lane_dispersions = dict(callback._lane_spreads)
    return callback


# What the 370M probe measured at step 80, the last reading before the held-out evaluator's
# forward pass contaminated one. Blocks 01 and 02 sit under the 5e-3 floor and the other
# fourteen are three to thirteen times over it.
PROBE_370M_STEP_80 = [
    0.01280, 0.00382, 0.00179, 0.00654, 0.00947, 0.01449, 0.02180, 0.02423,
    0.02277, 0.02145, 0.01974, 0.01818, 0.01944, 0.02166, 0.02229, 0.02455,
]  # fmt: skip


def training_step(callback, model, batch=None, micro_batches: int = 1):
    """
    A forward pass bracketed the way the trainer brackets one. The monitor only reads
    activations between `pre_step` and `post_train_batch`, so a bare `model(...)` is invisible
    to it by design -- that is what keeps the evaluator's forward pass out of the metrics.
    """
    batch = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)) if batch is None else batch
    callback.pre_step({"input_ids": batch})
    for _ in range(micro_batches):
        out = model(batch)
    callback.post_train_batch()
    return out


def test_optim_group_overrides_split_static_from_dynamic():
    """
    "The static component does not utilize weight decay, whereas the dynamic component does."
    Both globs have to actually match, which build_groups checks for us -- it raises on a
    pattern that hits nothing.
    """
    hc = HyperConnectionConfig(n_lanes=4)
    model = build_model(hc)
    optim = AdamWConfig(
        lr=1e-3,
        weight_decay=WEIGHT_DECAY,
        group_overrides=hc.optim_group_overrides(weight_decay=WEIGHT_DECAY),
    )

    groups = optim.build_groups(model)
    by_decay = {g.get("weight_decay", WEIGHT_DECAY): g for g in groups}

    static_count = sum(1 for n, _ in model.named_parameters() if "hc_static_" in n)
    dynamic_count = sum(1 for n, _ in model.named_parameters() if "hc_dynamic_" in n)
    assert static_count == 2 * 2 * 3  # 2 blocks x 2 streams x (B, A_m, A_r)
    assert dynamic_count == 2 * 2 * 5  # ... x (W_beta, W_m, W_r, s_beta, s_alpha)

    assert len(by_decay[0.0]["params"]) == static_count
    assert sum(1 for g in groups if g.get("weight_decay") == WEIGHT_DECAY) == 1


def test_optim_group_overrides_omit_the_dynamic_group_for_shc():
    hc = HyperConnectionConfig(n_lanes=4, dynamic=False)
    overrides = hc.optim_group_overrides(weight_decay=WEIGHT_DECAY)
    assert len(overrides) == 1
    AdamWConfig(lr=1e-3, group_overrides=overrides).build_groups(build_model(hc))


def test_monitor_records_the_four_diagnostics():
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model)

    training_step(callback, model)
    callback.pre_optim_step()

    metrics = callback.trainer.metrics  # type: ignore[attr-defined]
    for lane in range(4):
        assert f"hc/block 00/lane {lane} norm" in metrics
    assert "hc/block 01/hidden norm" in metrics
    assert "hc/block 00/rho(A_r) attention" in metrics
    assert "hc/block 00/rho(A_r) feed_forward" in metrics
    assert "hc/composite condition number" in metrics
    assert "hc/block 00/lane dispersion" in metrics
    assert "hc/min lane norm spread" in metrics
    assert "hc/min lane dispersion" in metrics
    assert "hc/median lane dispersion" in metrics
    assert "hc/differentiated block fraction" in metrics

    # A_r is the identity at init, so every radius and the composite condition are exactly 1.
    assert metrics["hc/block 00/rho(A_r) attention"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["hc/composite condition number"] == pytest.approx(1.0, abs=1e-4)


def test_doubly_stochastic_pins_the_spectral_radius_at_one():
    """
    mHC's claim is that a Birkhoff-constrained mixing matrix keeps the composite across depth
    well conditioned however far the parameters drift. Perturb them and check it holds.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4, doubly_stochastic=True))
    callback = attach(model)

    with torch.no_grad():
        for block in model.blocks.values():
            for stream in (block.attention_residual_stream, block.feed_forward_residual_stream):
                stream.hc_static_alpha_r.add_(torch.randn_like(stream.hc_static_alpha_r))

    training_step(callback, model)
    callback.pre_optim_step()

    metrics = callback.trainer.metrics  # type: ignore[attr-defined]
    assert metrics["hc/block 00/rho(A_r) attention"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["hc/composite spectral radius"] == pytest.approx(1.0, abs=1e-3)


def test_fail_closed_when_the_lanes_never_differentiate():
    """
    At initialization the lanes are identical by construction, which is exactly the state the
    rehearsal must refuse to train through. Every block reads exactly zero on both statistics,
    so no aggregation over blocks can rescue it.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model, fail_closed_by_step=0)

    training_step(callback, model)
    assert callback.trainer.metrics["hc/block 00/lane dispersion"] == 0.0  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="lanes are still identical"):
        callback.pre_optim_step()
    assert callback.trainer.metrics["hc/differentiated block fraction"] == 0.0  # type: ignore[attr-defined]


def test_lane_dispersion_sees_lanes_that_the_norm_spread_calls_identical():
    """
    Four lanes related by a rotation hold four different vectors of one length. The spread of
    the per-lane norms is exactly zero on them and says the mechanism is inert; it is measuring
    length, and length is the one thing a rotation preserves. Dispersion about the lane mean is
    zero only when the lanes really are one vector, which is what the guard claims to test.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model)

    rotation, _ = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL))
    lane = torch.randn(2, SEQ_LEN, 1, D_MODEL)
    lanes = torch.cat([lane @ torch.linalg.matrix_power(rotation, k) for k in range(4)], dim=-2)

    callback.pre_step({"input_ids": None})
    callback._activation_hook(None, None, lanes, block_name="blocks.0")
    callback.post_train_batch()

    metrics = callback.trainer.metrics  # type: ignore[attr-defined]
    assert metrics["hc/block 00/lane norm spread"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hc/block 00/lane dispersion"] > 1.0


def test_dispersion_is_never_below_the_norm_spread():
    """
    The floor was calibrated against the norm spread and is now applied to dispersion, which is
    only sound because dispersion dominates it: mean_k||x_k - x_bar||^2 = mean_k||x_k||^2 -
    ||x_bar||^2 and ||x_bar|| is at most the mean lane norm. So no block that used to clear the
    floor is refused by the new statistic.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model)

    with torch.no_grad():
        for block in model.blocks.values():
            for stream in (block.attention_residual_stream, block.feed_forward_residual_stream):
                stream.hc_static_beta.add_(torch.randn_like(stream.hc_static_beta) * 0.3)

    training_step(callback, model)
    metrics = callback.trainer.metrics  # type: ignore[attr-defined]
    for label in ("block 00", "block 01"):
        assert metrics[f"hc/{label}/lane dispersion"] >= metrics[f"hc/{label}/lane norm spread"]


def test_two_flat_blocks_out_of_sixteen_do_not_veto_the_run(caplog):
    """
    The 370M probe's step-80 reading. Blocks 01 and 02 never separated while the other fourteen
    did, and a minimum over blocks would have killed a 56-hour run on the strength of the single
    block that showed the least. The run stands, and the two blocks are named in the log.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model, fail_closed_by_step=0)
    measured_blocks(callback, PROBE_370M_STEP_80)

    with caplog.at_level("WARNING"):
        callback._check_lanes_differentiated()

    metrics = callback.trainer.metrics  # type: ignore[attr-defined]
    assert metrics["hc/differentiated block fraction"] == pytest.approx(14 / 16)
    assert metrics["hc/min lane norm spread"] == pytest.approx(0.00179)
    assert metrics["hc/median lane dispersion"] == pytest.approx(0.01959, abs=1e-5)
    assert callback.fail_closed_by_step is None
    assert "blocks.1" in caplog.text and "blocks.2" in caplog.text


@pytest.mark.parametrize("n_flat", [0, 4, 8])
def test_the_rule_stands_while_a_majority_of_blocks_differentiate(n_flat):
    callback = attach(build_model(HyperConnectionConfig(n_lanes=4)), fail_closed_by_step=0)
    measured_blocks(callback, [1e-4] * n_flat + [3e-2] * (16 - n_flat))

    callback._check_lanes_differentiated()
    assert callback.fail_closed_by_step is None


@pytest.mark.parametrize("n_flat", [9, 15, 16])
def test_the_rule_fails_once_a_majority_of_blocks_are_flat(n_flat):
    """
    A mechanism that only works in a handful of layers is the failure this guard exists for, and
    it has to survive the aggregation that stops one shallow block from vetoing on its own.
    """
    callback = attach(build_model(HyperConnectionConfig(n_lanes=4)), fail_closed_by_step=0)
    measured_blocks(callback, [1e-4] * n_flat + [3e-2] * (16 - n_flat))

    with pytest.raises(RuntimeError, match="lanes are still identical"):
        callback._check_lanes_differentiated()


def test_the_floor_is_not_reached_before_the_step_it_is_set_for():
    callback = attach(build_model(HyperConnectionConfig(n_lanes=4)), fail_closed_by_step=100)
    measured_blocks(callback, [0.0] * 16)

    callback._check_lanes_differentiated()
    assert callback.fail_closed_by_step == 100
    assert callback.trainer.metrics["hc/differentiated block fraction"] == 0.0  # type: ignore[attr-defined]


def test_fail_closed_passes_once_the_lanes_differentiate():
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model, fail_closed_by_step=0)

    with torch.no_grad():
        for block in model.blocks.values():
            for stream in (block.attention_residual_stream, block.feed_forward_residual_stream):
                stream.hc_static_beta.add_(torch.randn_like(stream.hc_static_beta) * 0.3)

    training_step(callback, model)
    callback.pre_optim_step()

    assert callback.fail_closed_by_step is None
    assert callback.trainer.metrics["hc/min lane norm spread"] > 1e-3  # type: ignore[attr-defined]


def test_monitor_refuses_a_model_with_no_hyper_connections():
    from olmo_core.exceptions import OLMoConfigurationError

    config = TransformerConfig.llama_like(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=2, n_heads=4
    )
    model = config.build()
    callback = HyperConnectionMonitorCallback()
    callback.trainer = FakeTrainer(FakeTrainModule(model))  # type: ignore[assignment]
    with pytest.raises(OLMoConfigurationError, match="no hyper-connection blocks"):
        callback.post_attach()


def test_monitor_only_measures_on_the_interval():
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = HyperConnectionMonitorCallback(interval=50)
    callback.trainer = FakeTrainer(FakeTrainModule(model), global_step=7)  # type: ignore[assignment]
    callback.post_attach()
    callback.pre_train()

    training_step(callback, model)
    callback.pre_optim_step()

    assert callback.trainer.metrics == {}  # type: ignore[attr-defined]


def test_the_monitor_ignores_forward_passes_that_are_not_training_steps():
    """
    An evaluator runs a forward pass in post_step, after post_train_batch. The forward hook
    fires on it too, and it is over held-out padded sequences rather than the training batch.
    In the rehearsal that understated lane spread by 11-50%, worst at the last step of the run
    -- which is the value that lands in the run summary.

    Gating on the training step rather than on `module.training`, because the evaluator's own
    comment says it means to switch the model to eval mode and the line is commented out.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model)
    batch = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    # A forward pass before any training step -- eval_on_startup does exactly this.
    model(batch)
    assert callback.trainer.metrics == {}, "captured a forward pass outside a training step"  # type: ignore[attr-defined]

    callback.pre_step({"input_ids": batch})
    model(batch)
    callback.post_train_batch()
    captured = dict(callback.trainer.metrics)  # type: ignore[attr-defined]
    assert captured, "did not capture the training step"

    # The evaluator's forward pass, after post_train_batch. Must change nothing.
    model(batch * 0 + 1)
    assert callback.trainer.metrics == captured  # type: ignore[attr-defined]


def test_gradient_accumulation_does_not_drop_the_lane_metrics():
    """
    A forward hook fires once per micro-batch, and the rehearsal runs eight of them per step.
    Without a merge strategy the trainer warns on each duplicate and keeps the first value, so
    the guard would be reading the first micro-batch of the step and the log would carry a few
    hundred warnings per logged step.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model)

    training_step(callback, model, micro_batches=8)
    callback.pre_optim_step()

    assert "hc/block 00/lane 0 norm" in callback.trainer.metrics  # type: ignore[attr-defined]
    assert "hc/composite condition number" in callback.trainer.metrics  # type: ignore[attr-defined]


def test_reduce_type_is_stable_for_every_metric_across_steps():
    """
    The trainer raises if a metric name is ever recorded with a different reduce type than the
    one it was first seen with, and it raises mid-run rather than at construction.
    """
    model = build_model(HyperConnectionConfig(n_lanes=4))
    callback = attach(model)

    for step in range(3):
        callback.trainer.global_step = step  # type: ignore[attr-defined]
        callback.trainer.seen_this_step.clear()  # type: ignore[attr-defined]
        training_step(callback, model)
        callback.pre_optim_step()

    assert len(callback.trainer.reduce_types) > 10  # type: ignore[attr-defined]


def test_metric_names_are_stable_across_layers():
    model = build_model(HyperConnectionConfig(n_lanes=2))
    callback = attach(model)
    training_step(callback, model)
    callback.pre_optim_step()

    labels: List[str] = sorted(
        {k.split("/")[1] for k in callback.trainer.metrics if k.startswith("hc/block")}  # type: ignore[attr-defined]
    )
    assert labels == ["block 00", "block 01"]
