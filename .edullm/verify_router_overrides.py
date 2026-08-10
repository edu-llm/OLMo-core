"""Prove the throughput settings reach the fields they name, before a dispatch spends a queue.

    python .edullm/verify_router_overrides.py

Exits 0 when every override lands and non-zero on the first that does not. Reaches no network,
needs no GPU, and takes under a second, so it can be a gate rather than something somebody reads.

WHY THIS EXISTS. ``ExperimentConfig.merge`` sets fields by dotted path and a path that names
nothing raises -- but only for the spelling, not for the meaning. The failure this is guarding
against is subtler than a typo: ``bias_gamma`` is ``Optional[float] = None`` and the router only
registers its ``score_bias`` buffer when the value is not None *at build time*, so an override
applied after the model is built would leave the field set, the log honest, and the mechanism
absent. The three paths below are the ones the MFU report recommends, and each is checked all the
way through to the built module rather than to the config that describes it.

THE ONE THAT IS NOT A THROUGHPUT LEVER IS THE MOST IMPORTANT LINE HERE. Turning ``bias_gamma`` on
adds ``blocks.*.feed_forward_moe.router.score_bias`` to the state dict, and turning it off removes
it. A checkpoint written by one and resumed by the other does not agree, and the run that finds
out is the one that has already spent nine hours. The last check pins that both ways round so the
decision is known to be a before-step-one decision rather than discovered to be one.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig

#: The run's shape, from ``olmoe_7b_32x4`` in ``train_on_corpus.py``, at one layer instead of
#: sixteen. The overrides under test address ``block.*``, which every layer shares, so a second
#: layer would check the same field twice and cost sixteen times the build.
SHAPE: dict[str, Any] = dict(
    vocab_size=100_352,
    d_model=2048,
    n_layers=1,
    n_heads=16,
    num_experts=32,
    top_k=4,
    expert_hidden_size=2048,
    dropless=True,
    lb_loss_weight=0.01,
    z_loss_weight=0.001,
    reordered_norm=True,
    qk_norm=True,
    rope_theta=500_000,
    layer_norm_eps=1e-6,
)

#: What the report tells somebody to append to ``command:``. Written here exactly as it would be
#: typed, prefix included, so that a rename upstream fails this file rather than a dispatch.
MODEL_PREFIX = "model."
ROUTER_BIAS_GAMMA = "model.block.feed_forward_moe.router.bias_gamma"
ROUTER_UNIFORM = "model.block.feed_forward_moe.router.uniform_expert_assignment"
DP_PREFETCH = "train_module.dp_config.prefetch_factor"


def _model_config() -> TransformerConfig:
    return TransformerConfig.llama_like_moe(**SHAPE)


def _merge(config: TransformerConfig, override: str) -> TransformerConfig:
    """Apply an override written with its ``model.`` prefix to a bare model config."""
    assert override.startswith(MODEL_PREFIX), override
    return config.merge([override[len(MODEL_PREFIX) :]])


def check(name: str, got: Any, want: Any) -> bool:
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f" (wanted {want!r})"))
    return ok


def main() -> int:
    failures = 0

    # 1. The field the report calls the largest lever exists, defaults off, and takes a float.
    base = _model_config()
    failures += not check(
        "bias_gamma defaults off", base.block.feed_forward_moe.router.bias_gamma, None
    )

    biased = _merge(base, f"{ROUTER_BIAS_GAMMA}=0.0001")
    failures += not check(
        "bias_gamma override lands", biased.block.feed_forward_moe.router.bias_gamma, 0.0001
    )

    # 2. It survives the build. A config field that never reaches a module is the failure mode
    #    this whole file is about -- the router reads `bias_gamma` in `__init__` and registers
    #    `score_bias` there, so a value that arrives late changes nothing while looking applied.
    with torch.device("meta"):
        biased_model = biased.build(init_device="meta")
    router = biased_model.blocks["0"].feed_forward_moe.router
    failures += not check("built router carries gamma", router.bias_gamma, 0.0001)
    failures += not check(
        "built router carries the bias buffer",
        None if router.score_bias is None else tuple(router.score_bias.shape),
        (SHAPE["num_experts"],),
    )

    # 3. The benchmarking flag that measures what balance is worth, which is the same override
    #    machinery and is the experiment the report asks for before item 2 is taken.
    uniform = _merge(base, f"{ROUTER_UNIFORM}=true")
    failures += not check(
        "uniform_expert_assignment override lands",
        uniform.block.feed_forward_moe.router.uniform_expert_assignment,
        True,
    )
    with torch.device("meta"):
        uniform_model = uniform.build(init_device="meta")
    failures += not check(
        "built router routes uniformly",
        uniform_model.blocks["0"].feed_forward_moe.router.uniform_expert_assignment,
        True,
    )

    # 4. FSDP forward prefetch, which is pure scheduling and cannot move a number. Built the way
    #    `train_on_corpus.build_config` builds it for this model, HSDP and mesh included, because
    #    the field being checked is the one that config will carry.
    dp = TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        num_replicas=8,
        shard_degree=8,
    )
    failures += not check("prefetch_factor defaults to none", dp.prefetch_factor, 0)
    failures += not check(
        "prefetch_factor override lands",
        dp.merge([DP_PREFETCH[len("train_module.dp_config.") :] + "=2"]).prefetch_factor,
        2,
    )

    # 5. The checkpoint hazard, both ways round.
    with torch.device("meta"):
        plain_model = base.build(init_device="meta")
    plain_keys = {k for k in plain_model.state_dict() if k.endswith("router.score_bias")}
    biased_keys = {k for k in biased_model.state_dict() if k.endswith("router.score_bias")}
    failures += not check("without gamma the state dict has no bias", plain_keys, set())
    failures += not check(
        "with gamma it has one per block",
        biased_keys,
        {"blocks.0.feed_forward_moe.router.score_bias"},
    )

    if failures:
        print(f"\n{failures} check(s) failed. Do not dispatch against these overrides.")
        return 1
    print("\nEvery override reaches the module it names.")
    print(
        "Reminder: `score_bias` is in the state dict only when gamma is on, so this is a\n"
        "decision to take before step one and not at a resume."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
