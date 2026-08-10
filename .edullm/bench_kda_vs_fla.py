"""Measure what is actually available to make the ``kda`` arm faster.

Three questions, in the order they matter, all at the arm's geometry.

**Which kernel backend is even reachable.** ``fla.ops.kda.chunk_kda`` carries ``@dispatch('kda')``
and the registry holds two alternatives to the Triton path: ``FlashKDABackend`` needs the
``flash_kda`` package and ``KDATileLangBackend`` needs ``tilelang``. Neither is installed by
``.edullm/Dockerfile``, so both should report absent -- and confirming that from inside the image
is worth more than reading the Dockerfile, because the dispatch is on the op both this repository
and upstream call, which means adding a package would move every KDA arm at once.

**What the kernel's own opt-in flags are worth.** The arm calls ``chunk_kda`` with
``use_qk_l2norm_in_kernel`` and ``use_gate_in_kernel`` and nothing else. Four more exist and all
default off, so upstream's layer does not use them either and a layer-versus-layer comparison
cannot see them. ``disable_recompute`` and ``state_v_first`` and ``use_beta_sigmoid_in_kernel``
are numerically free; ``safe_gate`` is NOT -- with ``lower_bound`` it replaces the gate
``-exp(A_log) * softplus(g + dt_bias)`` with ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``,
a different model that would need its own cell rather than a swap into a frozen arm. It is
measured here anyway, because knowing the size of the prize is what decides whether that cell is
worth running, and it is reported with ``changes_numerics`` set so no reader mistakes it.

**Whether the layer wrapper costs anything.** This repository's ``KimiDeltaAttention`` is a port
of ``fla.layers.kda`` and a source diff shows the same operator sequence through the same three
fused kernels. Upstream's weights are copied onto the port so both time the same function, and
the maximum absolute output difference is reported beside the timings as a parity receipt.

The short convolutions get their own comparison because the arm hardcodes ``backend="triton"``
and ``fla.modules.conv.cuda`` holds a CUDA path the arm has never tried.
"""

import argparse
import importlib
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
#: The arm names no ``norm_eps`` and takes the config default, which upstream also defaults to.
NORM_EPS = 1e-5
SEQUENCE_LENGTH = 4096

#: Two is the arm's own per-rank microbatch of 8,192 tokens at sequence length 4,096. Eight is
#: the same work with launch overhead amortized, where a kernel difference shows largest.
BATCH_SIZES = (2, 8)

#: What the arm passes ``chunk_kda`` today. Every variant below is this plus something.
BASELINE_KWARGS: dict[str, Any] = {
    "use_qk_l2norm_in_kernel": True,
    "use_gate_in_kernel": True,
}

#: ``(name, extra kwargs, whether it changes what the model computes)``.
KNOB_VARIANTS: list[tuple[str, dict[str, Any], bool]] = [
    ("baseline", {}, False),
    ("beta_sigmoid_in_kernel", {"use_beta_sigmoid_in_kernel": True}, False),
    ("state_v_first", {"state_v_first": True}, False),
    ("disable_recompute", {"disable_recompute": True}, False),
    (
        "free_flags_together",
        {
            "use_beta_sigmoid_in_kernel": True,
            "state_v_first": True,
            "disable_recompute": True,
        },
        False,
    ),
    ("safe_gate_lower_bound_5", {"safe_gate": True, "lower_bound": -5.0}, True),
    (
        "everything",
        {
            "use_beta_sigmoid_in_kernel": True,
            "state_v_first": True,
            "disable_recompute": True,
            "safe_gate": True,
            "lower_bound": -5.0,
        },
        True,
    ),
]


def backend_availability() -> dict[str, Any]:
    """Which alternative KDA backends this image can reach, asked of the image itself."""
    found: dict[str, Any] = {}
    for package in ("flash_kda", "tilelang"):
        try:
            importlib.import_module(package)
            found[package] = "present"
        except Exception as error:
            found[package] = f"absent ({type(error).__name__})"
    try:
        from fla.ops.kda.backends import kda_registry

        found["kda_registry"] = repr(kda_registry)
    except Exception as error:
        found["kda_registry"] = f"unreadable: {type(error).__name__}: {error}"
    return found


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
    """Build upstream's KDA layer at the same geometry and dtype, at library defaults."""
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


def summarize(timings: list[float], tokens: int) -> dict[str, float]:
    """Median, p90 and the token rate the median implies."""
    timings.sort()
    median = statistics.median(timings)
    return {
        "ms_p50": median,
        "ms_p90": timings[min(len(timings) - 1, int(0.9 * len(timings)))],
        "tok_s": tokens / (median / 1000.0),
    }


def time_callable(
    run: Callable[[], torch.Tensor],
    reset: Callable[[], None],
    *,
    tokens: int,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    """Time forward plus backward of ``run``, calling ``reset`` before each step."""
    for _ in range(warmup):
        reset()
        run().float().pow(2).mean().backward()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(steps):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run().float().pow(2).mean().backward()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return summarize(timings, tokens)


def kernel_sweep(
    *, batch: int, seq_len: int, device: torch.device, dtype: torch.dtype, warmup: int, steps: int
) -> dict[str, Any]:
    """Time ``chunk_kda`` at the arm's shapes under each combination of its opt-in flags."""
    from fla.ops.kda import chunk_kda

    generator = torch.Generator(device=device).manual_seed(20260810)

    def randn(*shape: int, requires_grad: bool = True) -> torch.Tensor:
        tensor = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
        return tensor.requires_grad_(requires_grad)

    q = randn(batch, seq_len, N_HEADS, HEAD_DIM)
    k = randn(batch, seq_len, N_HEADS, HEAD_DIM)
    v = randn(batch, seq_len, N_HEADS, HEAD_DIM)
    g = randn(batch, seq_len, N_HEADS, HEAD_DIM)
    beta_raw = randn(batch, seq_len, N_HEADS)
    # Uniform on (1, 16) then log, which is the mixer's own A_log range. Starting at zero would
    # make exp(A_log) a zero decay and time a recurrence that never changes its state.
    a_log = (
        (torch.rand(N_HEADS, device=device, dtype=torch.float32, generator=generator) * 15.0 + 1.0)
        .log()
        .requires_grad_(True)
    )
    dt_bias = torch.rand(
        N_HEADS * HEAD_DIM, device=device, dtype=torch.float32, generator=generator
    ).requires_grad_(True)
    leaves = [q, k, v, g, beta_raw, a_log, dt_bias]

    def reset() -> None:
        for leaf in leaves:
            leaf.grad = None

    results: dict[str, Any] = {}
    for name, extra, changes_numerics in KNOB_VARIANTS:
        kwargs = {**BASELINE_KWARGS, **extra}
        # The kernel applies the sigmoid itself only when asked to; otherwise beta arrives
        # already activated. Passing the raw tensor both ways would time two different betas.
        beta = beta_raw if kwargs.get("use_beta_sigmoid_in_kernel") else beta_raw.sigmoid()

        def run(kwargs: dict[str, Any] = kwargs, beta: torch.Tensor = beta) -> torch.Tensor:
            out, _ = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, A_log=a_log, dt_bias=dt_bias, **kwargs
            )
            return out

        try:
            measurement = time_callable(
                run, reset, tokens=batch * seq_len, warmup=warmup, steps=steps
            )
        except Exception as error:
            measurement = {"error": f"{type(error).__name__}: {error}"}
        measurement["changes_numerics"] = changes_numerics
        results[name] = measurement

    base = results.get("baseline", {})
    if "ms_p50" in base:
        for name, measurement in results.items():
            if "ms_p50" in measurement:
                measurement["speedup_vs_baseline"] = base["ms_p50"] / measurement["ms_p50"]
    return results


def convolution_sweep(
    *, batch: int, seq_len: int, device: torch.device, dtype: torch.dtype, warmup: int, steps: int
) -> dict[str, Any]:
    """Time one short convolution on each backend ``fla`` offers for it."""
    from fla.modules.convolution import causal_conv1d

    x = torch.randn(batch, seq_len, D_MODEL, device=device, dtype=dtype).requires_grad_(True)
    weight = torch.randn(D_MODEL, CONV_SIZE, device=device, dtype=dtype).requires_grad_(True)

    def reset() -> None:
        x.grad = None
        weight.grad = None

    results: dict[str, Any] = {}
    for backend in ("triton", "cuda"):

        def run(backend: str = backend) -> torch.Tensor:
            out = causal_conv1d(x=x, weight=weight, bias=None, activation="silu", backend=backend)
            return out[0] if isinstance(out, tuple) else out

        try:
            results[backend] = time_callable(
                run, reset, tokens=batch * seq_len, warmup=warmup, steps=steps
            )
        except Exception as error:
            results[backend] = {"error": f"{type(error).__name__}: {error}"}

    triton, cuda = results.get("triton", {}), results.get("cuda", {})
    if "ms_p50" in triton and "ms_p50" in cuda:
        results["cuda_speedup"] = triton["ms_p50"] / cuda["ms_p50"]
    return results


def port_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x)


def upstream_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x)[0]


def layer_comparison(
    *,
    port: torch.nn.Module,
    upstream: torch.nn.Module,
    x: torch.Tensor,
    compile_too: bool,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    """Time the port against upstream's layer, eager and compiled, on identical weights."""
    variants: list[tuple[str, torch.nn.Module, Callable]] = [
        ("upstream_eager", upstream, upstream_forward),
        ("port_eager", port, port_forward),
    ]
    if compile_too:
        variants += [
            ("upstream_compiled", torch.compile(upstream), upstream_forward),
            ("port_compiled", torch.compile(port), port_forward),
        ]

    row: dict[str, Any] = {}
    try:
        with torch.no_grad():
            difference = (upstream_forward(upstream, x) - port_forward(port, x)).abs().max()
        row["max_abs_output_diff"] = float(difference)
    except Exception as error:
        row["max_abs_output_diff"] = f"unmeasured: {type(error).__name__}: {error}"

    for name, module, call in variants:
        try:
            row[name] = time_callable(
                lambda module=module, call=call: call(module, x),
                lambda module=module: module.zero_grad(set_to_none=True),
                tokens=x.shape[0] * x.shape[1],
                warmup=warmup,
                steps=steps,
            )
        except Exception as error:
            row[name] = {"error": f"{type(error).__name__}: {error}"}

    for mode in ("eager", "compiled"):
        up, own = row.get(f"upstream_{mode}"), row.get(f"port_{mode}")
        if not (isinstance(up, dict) and isinstance(own, dict)):
            continue
        if "ms_p50" in up and "ms_p50" in own:
            row[f"upstream_speedup_{mode}"] = own["ms_p50"] / up["ms_p50"]
    return row


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
    parser.add_argument("--no-layer-comparison", action="store_true")
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
        "backends": backend_availability(),
        "kernel_sweep": {},
        "convolution_sweep": {},
        "layer_comparison": {},
    }
    try:
        from importlib.metadata import version

        report["fla"] = version("flash-linear-attention")
    except Exception as error:
        report["fla"] = f"unreadable: {error}"

    for batch in options.batch_sizes:
        key = f"batch_{batch}"
        report["kernel_sweep"][key] = kernel_sweep(
            batch=batch,
            seq_len=options.seq_len,
            device=device,
            dtype=dtype,
            warmup=options.warmup,
            steps=options.steps,
        )
        report["convolution_sweep"][key] = convolution_sweep(
            batch=batch,
            seq_len=options.seq_len,
            device=device,
            dtype=dtype,
            warmup=options.warmup,
            steps=options.steps,
        )

    if not options.no_layer_comparison:
        port = build_port(device, dtype)
        try:
            upstream = build_upstream(device, dtype)
        except Exception as error:
            report["layer_comparison"] = {
                "error": f"upstream layer would not build: {type(error).__name__}: {error}"
            }
        else:
            report["weight_copy"] = copy_weights(upstream, port)
            for batch in options.batch_sizes:
                x = torch.randn(batch, options.seq_len, D_MODEL, device=device, dtype=dtype)
                report["layer_comparison"][f"batch_{batch}"] = layer_comparison(
                    port=port,
                    upstream=upstream,
                    x=x,
                    compile_too=not options.no_compile,
                    warmup=options.warmup,
                    steps=options.steps,
                )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
