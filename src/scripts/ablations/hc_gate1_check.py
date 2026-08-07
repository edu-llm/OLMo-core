"""
GATE_1_CORRECTNESS for hyper-connections, on CPU.

The mHC dossier puts three gates in front of a distributed MoE implementation and says not to
advance past a failing one. This script is the first:

    GATE_1_CORRECTNESS:
        Baseline-equivalent initialization, constrained matrices, gradients,
        save/resume, and eager/compile parity.

It runs entirely on CPU, touches no network and trains nothing. Every check prints PASS or FAIL
with the number it was decided on, and the script exits nonzero if any of them fails::

    python src/scripts/ablations/hc_gate1_check.py
    python src/scripts/ablations/hc_gate1_check.py --check-compile

Eager/compile parity is opt-in because ``torch.compile`` takes minutes on a CPU; without
``--check-compile`` it is reported as SKIP rather than quietly counted as a pass.
"""

import argparse
import copy
import dataclasses
import logging
import sys
from typing import Callable, List, Optional, Tuple, cast

import torch

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.hyper_connections import (
    HyperConnection,
    HyperConnectionConfig,
    ResidualMixerType,
    StreamCollapseConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer import (
    HyperConnectionTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

MIXERS = list(ResidualMixerType)
DOUBLY_STOCHASTIC_MIXERS = [m for m in MIXERS if m != ResidualMixerType.unconstrained]
N_STREAMS = 4
D_MODEL = 64
VOCAB_SIZE = 256

#: A check's outcome: its name, whether it passed, and the number it was decided on.
CheckResult = Tuple[str, Optional[bool], str]


def _hc_block(
    mixer: ResidualMixerType, *, init_noise_std: float = 0.0, dropout_p: float = 0.0
) -> HyperConnectionTransformerBlock:
    return HyperConnectionTransformerBlock(
        d_model=D_MODEL,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(n_heads=4),
        feed_forward=FeedForwardConfig(hidden_size=2 * D_MODEL),
        layer_norm=LayerNormConfig(),
        hyper_connection=HyperConnectionConfig(
            n_streams=N_STREAMS,
            mixer=mixer,
            init_noise_std=init_noise_std,
            residual_dropout_p=dropout_p,
        ),
        init_device="cpu",
    )


def _baseline_block() -> ReorderedNormTransformerBlock:
    return ReorderedNormTransformerBlock(
        d_model=D_MODEL,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(n_heads=4),
        feed_forward=FeedForwardConfig(hidden_size=2 * D_MODEL),
        layer_norm=LayerNormConfig(),
        init_device="cpu",
    )


def _base_model_config() -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=D_MODEL,
        n_layers=2,
        n_heads=4,
        vocab_size=VOCAB_SIZE,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        layer_norm_eps=1e-6,
        hidden_size_multiplier=1.5,
    )


def _hc_model_config(mixer: ResidualMixerType, *, init_noise_std: float = 0.0):
    base = _base_model_config()
    block = dataclasses.replace(
        cast(TransformerBlockConfig, base.block),
        name=TransformerBlockType.hyper_connection,
        hyper_connection=HyperConnectionConfig(
            n_streams=N_STREAMS,
            mixer=mixer,
            init_noise_std=init_noise_std,
            residual_dropout_p=0.0,
        ),
    )
    return dataclasses.replace(
        base,
        block=block,
        name=TransformerType.hyper_connection,
        stream_collapse=StreamCollapseConfig(n_streams=N_STREAMS),
    )


def check_baseline_equivalent_init() -> List[CheckResult]:
    """
    A hyper-connected model must, at initialisation with the symmetry-breaking noise off,
    produce the same logits as the ordinary backbone it wraps.

    This is the check that decides whether the whole thing is a reparameterisation of the
    baseline at step zero, which is the only way an ablation's arms start from a common point.

    :returns: One result per mixer.
    """
    results: List[CheckResult] = []
    seed_all(17)
    baseline = _base_model_config().build()
    baseline.init_weights()
    baseline.eval()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    with torch.no_grad():
        expected = baseline(input_ids)

    for mixer in MIXERS:
        model = _hc_model_config(mixer).build()
        model.init_weights()
        model.eval()
        _, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
        with torch.no_grad():
            delta = (model(input_ids) - expected).abs().max().item()
        ok = not unexpected and delta < 1e-5
        results.append(
            (
                f"baseline-equivalent init [{mixer}]",
                ok,
                f"max|logits - baseline| = {delta:.3e} (tol 1e-05), "
                f"{len(unexpected)} unexpected state-dict keys",
            )
        )
    return results


def check_constrained_matrices() -> List[CheckResult]:
    """
    Every constrained mixer must produce a nonnegative ``H_res`` with unit row and column sums,
    away from initialisation as well as at it, and the unconstrained mixer must not.

    :returns: One result per mixer.
    """
    results: List[CheckResult] = []
    for mixer in MIXERS:
        seed_all(23)
        hc = HyperConnectionConfig(
            n_streams=N_STREAMS, mixer=mixer, init_noise_std=0.5, residual_dropout_p=0.0
        ).build()
        with torch.no_grad():
            if hc.h_res_logits is not None:
                hc.h_res_logits.normal_(mean=0.0, std=2.0)
        h_res = hc.residual_mixer()
        row_err = (h_res.sum(dim=-1) - 1).abs().max().item()
        col_err = (h_res.sum(dim=-2) - 1).abs().max().item()
        min_entry = h_res.min().item()

        if mixer == ResidualMixerType.unconstrained:
            # The control arm is supposed to be off the manifold. Reporting it as a pass only
            # when it *is* off keeps a future refactor from turning the control into a
            # sixth constrained variant without anybody noticing.
            ok = max(row_err, col_err) > 1e-3
            detail = (
                f"unconstrained by design: row err {row_err:.3e}, col err {col_err:.3e}, "
                f"min entry {min_entry:+.3e}"
            )
        else:
            ok = row_err < 1e-5 and col_err < 1e-5 and min_entry >= 0.0
            detail = (
                f"row err {row_err:.3e}, col err {col_err:.3e}, min entry {min_entry:+.3e} "
                f"(tol 1e-05)"
            )
        results.append((f"constrained H_res [{mixer}]", ok, detail))
    return results


def check_gradient_flow() -> List[CheckResult]:
    """
    Gradients must reach every routing parameter and be finite.

    The loss is deliberately not a plain sum over streams: a doubly stochastic ``H_res``
    preserves the sum over streams exactly, so ``out.sum()`` has *zero* gradient with respect to
    the ``birkhoff`` and ``kronecker`` parameters and this check would report a failure that is
    not there.

    :returns: One result per mixer.
    """
    results: List[CheckResult] = []
    for mixer in MIXERS:
        seed_all(7)
        block = _hc_block(mixer, init_noise_std=1e-2)
        out = block(torch.randn(2, 6, N_STREAMS, D_MODEL))
        out.square().mean().backward()

        routing = [
            (name, param)
            for name, param in block.named_parameters()
            if name.startswith(("attention_hc", "feed_forward_hc"))
        ]
        missing = [name for name, param in routing if param.grad is None]
        non_finite = [
            name
            for name, param in routing
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]
        zero = [
            name
            for name, param in routing
            if param.grad is not None and param.grad.abs().sum().item() == 0.0
        ]
        smallest = min(
            (param.grad.abs().max().item() for _, param in routing if param.grad is not None),
            default=0.0,
        )
        ok = bool(routing) and not missing and not non_finite and not zero
        results.append(
            (
                f"gradient flow [{mixer}]",
                ok,
                f"{len(routing)} routing tensors, smallest max|grad| {smallest:.3e}, "
                f"{len(missing)} missing, {len(non_finite)} non-finite, {len(zero)} all-zero",
            )
        )
    return results


def check_save_resume() -> List[CheckResult]:
    """
    A state-dict round-trip must reproduce the forward pass exactly, and must carry every
    routing parameter.

    :returns: One result per mixer.
    """
    results: List[CheckResult] = []
    for mixer in MIXERS:
        seed_all(13)
        block = _hc_block(mixer, init_noise_std=1e-2)
        block.eval()
        x = torch.randn(2, 5, N_STREAMS, D_MODEL)
        with torch.no_grad():
            before = block(x)

        state = copy.deepcopy(block.state_dict())
        restored = _hc_block(mixer, init_noise_std=1e-2)
        incompatible = restored.load_state_dict(state, strict=True)
        restored.eval()
        with torch.no_grad():
            after = restored(x)

        delta = (before - after).abs().max().item()
        n_routing_keys = len([k for k in state if "_hc." in k])
        expected_keys = 2 * (2 if mixer == ResidualMixerType.identity else 3)
        ok = (
            delta == 0.0
            and n_routing_keys == expected_keys
            and not incompatible.missing_keys
            and not incompatible.unexpected_keys
        )
        results.append(
            (
                f"save/resume [{mixer}]",
                ok,
                f"max|before - after| = {delta:.3e}, {n_routing_keys}/{expected_keys} routing "
                f"keys in the state dict",
            )
        )
    return results


def check_symmetry_breaking() -> List[CheckResult]:
    """
    The ``n`` streams must stay identical with the noise off and diverge with it on.

    Without this the :math:`S_n` permutation symmetry of the initialisation survives every
    gradient step, the streams stay copies of each other forever, and ``n > 1`` costs memory for
    nothing. Both directions are checked: a change that makes the streams diverge on their own
    would break the baseline equivalence above.

    :returns: One result per mixer.
    """
    results: List[CheckResult] = []
    for mixer in MIXERS:
        spreads = {}
        for label, noise in (("noise off", 0.0), ("noise on", 1e-2)):
            seed_all(3)
            block = _hc_block(mixer, init_noise_std=noise)
            optim = torch.optim.SGD(block.parameters(), lr=0.1)
            x = torch.randn(2, 6, D_MODEL)
            block(x).square().mean().backward()
            optim.step()
            out = block(x)
            spreads[label] = (out - out.mean(dim=-2, keepdim=True)).abs().max().item()

        ok = spreads["noise off"] < 1e-6 and spreads["noise on"] > 1e-5
        results.append(
            (
                f"symmetry breaking [{mixer}]",
                ok,
                f"stream spread after one step: {spreads['noise off']:.3e} with noise off "
                f"(want < 1e-06), {spreads['noise on']:.3e} with noise on (want > 1e-05)",
            )
        )
    return results


def check_float32_routing() -> List[CheckResult]:
    """
    Routing quantities must stay in float32 when the module is cast to bfloat16.

    :returns: One result per mixer.
    """
    results: List[CheckResult] = []
    for mixer in MIXERS:
        seed_all(5)
        hc = cast(
            HyperConnection,
            HyperConnectionConfig(
                n_streams=N_STREAMS, mixer=mixer, init_noise_std=1e-2, residual_dropout_p=0.0
            ).build(),
        ).to(torch.bfloat16)
        dtypes = {
            "h_pre": hc.read_in_gate().dtype,
            "h_post": hc.write_out_gate().dtype,
            "H_res": hc.residual_mixer().dtype,
        }
        out = hc(torch.randn(2, 4, N_STREAMS, D_MODEL, dtype=torch.bfloat16), lambda x: x)
        ok = all(d == torch.float32 for d in dtypes.values()) and out.dtype == torch.bfloat16
        results.append(
            (
                f"float32 routing in bf16 [{mixer}]",
                ok,
                ", ".join(f"{k}={v}" for k, v in dtypes.items()) + f", output={out.dtype}",
            )
        )
    return results


def check_eager_compile_parity(enabled: bool) -> List[CheckResult]:
    """
    ``torch.compile`` must not change the forward pass.

    :param enabled: Whether to actually run it. Compiling on CPU takes minutes, so this is
        opt-in and reported as a skip otherwise rather than silently counted as a pass.

    :returns: One result per mixer, or one skip per mixer.
    """
    if not enabled:
        return [
            (
                f"eager/compile parity [{mixer}]",
                None,
                "skipped; pass --check-compile to run it",
            )
            for mixer in MIXERS
        ]

    results: List[CheckResult] = []
    for mixer in MIXERS:
        seed_all(31)
        block = _hc_block(mixer, init_noise_std=1e-2)
        block.eval()
        x = torch.randn(2, 5, N_STREAMS, D_MODEL)
        with torch.no_grad():
            eager = block(x)
        try:
            compiled = torch.compile(block, fullgraph=False)
            with torch.no_grad():
                got = compiled(x)
            delta = (eager - got).abs().max().item()
            ok: Optional[bool] = delta < 1e-5
            detail = f"max|eager - compiled| = {delta:.3e} (tol 1e-05)"
        except Exception as exc:  # noqa: BLE001 - a gate reports the failure, it does not raise
            ok = False
            detail = f"torch.compile raised {exc!r}"
        results.append((f"eager/compile parity [{mixer}]", ok, detail))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run every GATE_1 check and print the verdict.

    :param argv: Arguments, defaulting to ``sys.argv[1:]``.

    :returns: 0 if every check that ran passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check-compile",
        action="store_true",
        help="also check eager/compile parity (slow on CPU)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    torch.manual_seed(0)

    groups: List[Tuple[str, Callable[[], List[CheckResult]]]] = [
        ("Baseline-equivalent initialisation", check_baseline_equivalent_init),
        ("Constrained matrices", check_constrained_matrices),
        ("Gradient flow", check_gradient_flow),
        ("Save / resume", check_save_resume),
        ("Symmetry breaking", check_symmetry_breaking),
        ("Float32 routing", check_float32_routing),
        ("Eager / compile parity", lambda: check_eager_compile_parity(args.check_compile)),
    ]

    print("\nGATE_1_CORRECTNESS - hyper-connections, CPU only\n")
    passed = failed = skipped = 0
    for title, run in groups:
        print(f"{title}")
        for name, ok, detail in run():
            if ok is None:
                status, skipped = "SKIP", skipped + 1
            elif ok:
                status, passed = "PASS", passed + 1
            else:
                status, failed = "FAIL", failed + 1
            print(f"  [{status}] {name:<42} {detail}")
        print()

    verdict = "GATE_1 PASSED" if failed == 0 else "GATE_1 FAILED"
    print(f"{verdict}: {passed} passed, {failed} failed, {skipped} skipped.")
    if skipped:
        print(
            "A skipped check is not a passed one. Nothing here says anything about FSDP, "
            "tensor\nor pipeline parallelism, MoE, or any behaviour under training; those are "
            "GATE_2 and\nGATE_3 and need GPUs."
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
