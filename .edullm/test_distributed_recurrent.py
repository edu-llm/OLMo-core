"""One end-to-end pass through the real training path: FSDP, activation checkpointing, optimizer.

Everything in the other two files exercises the model directly. This one goes through
``TransformerTrainModuleConfig.build`` and ``train_batch``, which is what the platform actually
runs, and it is where the interactions live that a bare forward cannot show: whether FSDP's
dynamically-created subclass still answers ``isinstance`` for the depth callback, whether
``n_loops`` survives the wrapping, and whether the three modules the recurrence adds get
gradients once they are DTensors rather than plain tensors.

ONE TEST, AND THAT IS A CONSTRAINT RATHER THAN A CHOICE. ``build_world_mesh`` raises "world
mesh already exists! You can only call 'build_world_mesh' once!" on a second call in the same
process, so a second train module here would fail on the harness rather than on anything real.
The single case turns activation checkpointing on, because FSDP with checkpointing is strictly
more machinery than FSDP alone.

Gloo on CPU at world size one, so it needs no GPU and no launcher.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo_core")


@pytest.fixture(scope="module")
def process_group():
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group already exists in this interpreter")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29677")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_the_real_training_path_wraps_runs_and_updates_the_recurrence(process_group):
    import olmo_recurrent as R
    from olmo_core.config import DType
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.nn.transformer import TransformerActivationCheckpointingMode as Mode
    from olmo_core.optim import AdamWConfig
    from olmo_core.train.train_module import (
        TransformerActivationCheckpointingConfig,
        TransformerDataParallelConfig,
        TransformerTrainModuleConfig,
    )

    torch.manual_seed(0)
    config = R.RecurrentTransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=6,
        n_heads=4,
        n_prelude=1,
        n_coda=1,
        default_n_loops=3,
        max_loops=3,
    ).apply_recurrent_residual_alpha()

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=64,
        max_sequence_length=32,
        optim=AdamWConfig(lr=1e-3),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.float32,
            reduce_dtype=DType.float32,
        ),
        ac_config=TransformerActivationCheckpointingConfig(mode=Mode.full),
        max_grad_norm=1.0,
    ).build(config.build(init_device="cpu"))

    model = train_module.model

    # FSDP swaps in a dynamically-created subclass. RecurrentDepthCallback gates on this
    # isinstance, so if it stopped holding the schedule would silently stop being applied.
    assert isinstance(model, R.RecurrentTransformer)
    assert list(model.blocks.keys()) == [str(i) for i in range(6)]
    assert model.n_loops == 3

    # The trainer is what metrics are recorded against, and building a real one would need a
    # data loader and a save folder for no extra coverage.
    class _Sink:
        global_step = 0
        max_steps = 100

        def record_metric(self, *args, **kwargs):
            pass

        def record_ce_loss(self, *args, **kwargs):
            pass

        def record_z_loss(self, *args, **kwargs):
            pass

        def record_loss(self, *args, **kwargs):
            pass

    train_module._trainer = _Sink()
    train_module.train_batch({"input_ids": torch.randint(0, 256, (4, 32))}, dry_run=False)
    train_module.optim_step()

    named = dict(model.named_parameters())
    for name in ("adapter.weight", "norm_e.weight", "injection.theta_A", "injection.B_cont"):
        grad = named[name].grad
        assert grad is not None, f"{name} got no gradient under FSDP"
        local = grad.to_local() if hasattr(grad, "to_local") else grad
        assert torch.isfinite(local).all(), f"{name} got a non-finite gradient under FSDP"

    # What the depth callback does between steps, on the wrapped model.
    model.n_loops = 2
    assert model.n_loops == 2
