"""Benchmark ``fla.layers.kda.KimiDeltaAttention`` against this repository's port of it.

The ``kda`` arm runs :class:`olmo_core.nn.attention.recurrent.KimiDeltaAttention`, a port of
upstream's layer written against ``fla`` v0.4.1. Both reach the same three fused kernels --
``fla.ops.kda.chunk_kda``, ``fla.modules.convolution.causal_conv1d`` and
``fla.modules.FusedRMSNormGated`` -- so a source diff already says the operator sequence is
identical. What a diff cannot settle is whether the wrapper around those kernels costs
anything measurable at the arm's geometry, and that is the only question this program answers.

Upstream v0.5.1 passes two kernel arguments the port does not: ``use_beta_sigmoid_in_kernel``
moves one elementwise sigmoid over a ``(B, T, n_heads)`` tensor inside the kernel, and
``state_v_first`` selects a state layout. The first is arithmetically negligible against q, k
and v at ``(B, T, 1024)``; the second is opaque from the outside. Whatever the two are worth
together shows up in the delta this program prints.

Weights are copied from the upstream layer onto the port where the mapping is unambiguous, so
both time the same function rather than two different random draws, and the maximum absolute
output difference is reported beside the timings as a parity receipt. A copy that fails is
recorded and the benchmark still runs -- a timing on differing weights is still a timing, and
losing it to a renamed attribute would be the worse outcome.
"""

import argparse
import json
import statistics
import sys
from typing import Any, Callable

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import KimiDeltaAttentionConfig

#: The ``kda`` arm's geometry, from ``.edullm/model_arch_tests.py``.
D_MODEL = 1024
N_HEADS = 16
HEAD_DIM = 64
CONV_SIZE = 4
#: The arm names no ``norm_eps`` and takes the config default, which upstream's layer also
#: defaults to. Passed explicitly to both here so the match is stated rather than assumed.
NORM_EPS = 1e-5
SEQUENCE_LENGTH = 4096

#: Batch sizes in sequences. Two is the arm's own per-rank microbatch of 8,192 tokens at
#: sequence length 4,096; eight is the same layer with enough work to hide launch overhead,
#: which is where a wrapper difference would be smallest and a kernel difference largest.
BATCH_SIZES = (2, 8)


def build_port(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Build this repository's KDA layer at the arm's exact configuration."""
    config = KimiDeltaAttentionConfig(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        norm_eps=NORM_EPS,
        dtype=DType.bfloat16 if dtype is torch.bfloat16 else DType.float32,
    )
    module = config.build(D_MODEL, layer_idx=0, n_layers=1, init_device=str(device))
    return module.to(device)


def build_upstream(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Build upstream's KDA layer at the same geometry and dtype on ``device``."""
    from fla.layers.kda import KimiDeltaAttention as UpstreamKDA

    module = UpstreamKDA(
        hidden_size=D_MODEL,
        head_dim=HEAD_DIM,
        num_heads=N_HEADS,
        mode="chunk",
        use_short_conv=True,
        conv_size=CONV_SIZE,
        conv_bias=False,
        norm_eps=NORM_EPS,
    )
    return module.to(device=device, dtype=dtype)


#: Upstream attribute -> port attribute, for every parameter the two layers share.
WEIGHT_MAP = {
    "q_proj.weight": "w_q.weight",
    "k_proj.weight": "w_k.weight",
    "v_proj.weight": "w_v.weight",
    "b_proj.weight": "w_b.weight",
    "f_proj.0.weight": "f_proj.0.weight",
    "f_proj.1.weight": "f_proj.1.weight",
    "g_proj.0.weight": "g_proj.0.weight",
    "g_proj.1.weight": "g_proj.1.weight",
    "g_proj.1.bias": "g_proj.1.bias",
    "q_conv1d.weight": "q_conv1d.weight",
    "k_conv1d.weight": "k_conv1d.weight",
    "v_conv1d.weight": "v_conv1d.weight",
    "o_norm.weight": "o_norm.weight",
    "o_proj.weight": "w_out.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
}


def copy_weights(upstream: torch.nn.Module, port: torch.nn.Module) -> str:
    """Copy every mapped parameter from ``upstream`` onto ``port``.

    :returns: ``"ok"``, or a sentence naming the first parameter that could not be copied.
    """
    source = dict(upstream.named_parameters())
    target = dict(port.named_parameters())
    with torch.no_grad():
        for upstream_name, port_name in WEIGHT_MAP.items():
            if upstream_name not in source:
                return f"upstream has no parameter '{upstream_name}'"
            if port_name not in target:
                return f"the port has no parameter '{port_name}'"
            src, dst = source[upstream_name], target[port_name]
            if src.shape != dst.shape:
                return (
                    f"'{upstream_name}' is {tuple(src.shape)} and "
                    f"'{port_name}' is {tuple(dst.shape)}"
                )
            dst.copy_(src.to(dst.dtype))
    unmapped = sorted(set(target) - set(WEIGHT_MAP.values()))
    if unmapped:
        return f"the port carries parameters upstream does not: {', '.join(unmapped)}"
    return "ok"


def port_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x)


def upstream_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x)[0]


def time_steps(
    module: torch.nn.Module,
    call: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    """Time forward plus backward, reporting the median and p90 of the timed steps."""
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        call(module, x).float().pow(2).mean().backward()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(steps):
        module.zero_grad(set_to_none=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call(module, x).float().pow(2).mean().backward()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    tokens = x.shape[0] * x.shape[1]
    median = statistics.median(timings)
    return {
        "ms_p50": median,
        "ms_p90": timings[min(len(timings) - 1, int(0.9 * len(timings)))],
        "tok_s": tokens / (median / 1000.0),
    }


def parity(upstream: torch.nn.Module, port: torch.nn.Module, x: torch.Tensor) -> float | None:
    """Maximum absolute difference between the two layers' outputs, or ``None`` if it raised."""
    try:
        with torch.no_grad():
            return float((upstream_forward(upstream, x) - port_forward(port, x)).abs().max())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    # NAMED ON THE COMMAND LINE RATHER THAN SET IN CODE. The platform's precision guard reads
    # the words of the command and cannot see a dtype a program chooses for itself, so a run
    # that only mentions this file is admitted onto a card with no bfloat16 in the hardware and
    # dies on the first kernel that needs it, after the machine has been billed.
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--no-compile", action="store_true")
    options = parser.parse_args()
    dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32

    if not torch.cuda.is_available():
        print(json.dumps({"error": "no CUDA device; this benchmark measures kernels"}))
        return 1

    device = torch.device("cuda")
    torch.manual_seed(4242)

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "head_dim": HEAD_DIM,
        "seq_len": options.seq_len,
        "dtype": options.dtype,
        "warmup_steps": options.warmup,
        "timed_steps": options.steps,
        "measurements": [],
    }
    try:
        from importlib.metadata import version

        report["fla"] = version("flash-linear-attention")
    except Exception as error:
        report["fla"] = f"unreadable: {error}"

    port = build_port(device, dtype)
    try:
        upstream = build_upstream(device, dtype)
    except Exception as error:
        print(
            json.dumps(
                {
                    **report,
                    "error": f"upstream layer would not build: {type(error).__name__}: {error}",
                }
            )
        )
        return 1

    report["weight_copy"] = copy_weights(upstream, port)

    variants: list[tuple[str, torch.nn.Module, Callable]] = [
        ("upstream_eager", upstream, upstream_forward),
        ("port_eager", port, port_forward),
    ]
    if not options.no_compile:
        variants += [
            ("upstream_compiled", torch.compile(upstream), upstream_forward),
            ("port_compiled", torch.compile(port), port_forward),
        ]

    for batch in options.batch_sizes:
        x = torch.randn(batch, options.seq_len, D_MODEL, device=device, dtype=dtype)
        row: dict[str, Any] = {
            "batch": batch,
            "tokens": batch * options.seq_len,
            "max_abs_output_diff": parity(upstream, port, x),
        }
        for name, module, call in variants:
            try:
                row[name] = time_steps(module, call, x, warmup=options.warmup, steps=options.steps)
            except Exception as error:
                row[name] = {"error": f"{type(error).__name__}: {error}"}
        for mode in ("eager", "compiled"):
            up, own = row.get(f"upstream_{mode}"), row.get(f"port_{mode}")
            if not (isinstance(up, dict) and isinstance(own, dict)):
                continue
            if "ms_p50" in up and "ms_p50" in own:
                row[f"upstream_speedup_{mode}"] = own["ms_p50"] / up["ms_p50"]
        report["measurements"].append(row)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
