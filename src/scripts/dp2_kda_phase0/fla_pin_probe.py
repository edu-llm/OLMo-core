"""Probe the two measured ``fla`` conventions the P0.3 external anchor depends on.

The anchor test ``test_r2_matches_external_gated_delta_product_naive`` was written against
``flash-linear-attention==0.5.1``, where two behaviours of
``fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product`` were *measured*:

1. **``scale`` is ignored** -- the argument is accepted but never applied to ``q``, so
   ``scale=1.0`` and ``scale=K**-0.5`` return byte-identical output.
2. **inputs are cast to float32** internally, so float64-ulp agreement is unattainable and the
   test must compare against a float32 floor it computes at run time.

``pyproject.toml`` pins ``flash-linear-attention==0.4.1``, so this script re-measures both
conventions at whatever version is installed in the active interpreter and prints a verdict. It is
a throwaway diagnostic, not part of the test suite.
"""

import inspect
import json
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F


def main() -> int:
    """Measure both conventions against the installed ``fla`` and print a JSON verdict.

    :returns: process exit status; ``0`` always, the verdict is in the printed JSON.
    """
    import fla  # type: ignore[import-not-found]
    from fla.ops.gated_delta_product.naive import (  # type: ignore[import-not-found]
        naive_recurrent_gated_delta_product,
    )

    out: Dict[str, Any] = {
        "fla_version": fla.__version__,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
    }

    src = inspect.getsource(naive_recurrent_gated_delta_product)
    out["signature"] = str(inspect.signature(naive_recurrent_gated_delta_product))
    # Convention 2, statically: does the body downcast to float32?
    out["source_has_float_cast"] = (
        "map(lambda x: x.float()" in src or ".float()" in src.split("\n")[1]
    )
    out["source_first_lines"] = [ln.strip() for ln in src.splitlines()[:24]]
    # Does `scale` appear anywhere in the body other than the signature/default assignment?
    body = src.split("\n", 1)[1]
    out["scale_mentions_in_body"] = [
        ln.strip() for ln in body.splitlines() if "scale" in ln and not ln.strip().startswith("#")
    ]

    device = torch.device("cuda")
    B, T, H, K, V, R = 1, 24, 2, 32, 32, 2
    gen = torch.Generator(device=device).manual_seed(101)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float64)

    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    beta = rnd(B, T * R, H).sigmoid()
    g_head = F.logsigmoid(rnd(B, T, H))
    g_channel = g_head[..., None].expand(B, T, H, K).contiguous()

    def call_fla(scale: float) -> torch.Tensor:
        o, _ = naive_recurrent_gated_delta_product(
            q=q,
            k=k,
            v=v,
            g=g_head,
            beta=beta,
            scale=scale,
            cu_seqlens=None,
            num_householder=R,
            output_final_state=True,
        )
        return o

    # --- Convention 1: is `scale` ignored? -------------------------------------------------
    o_scale1 = call_fla(1.0)
    o_scaled = call_fla(K**-0.5)
    same_bytes = bool(torch.equal(o_scale1, o_scaled))
    max_abs_gap = (o_scale1.double() - o_scaled.double()).abs().max().item()
    ratio = (o_scaled.double() / o_scale1.double().clamp_min(1e-30)).median().item()
    out["convention_1_scale_ignored"] = {
        "byte_identical": same_bytes,
        "max_abs_gap_scale1_vs_scaled": max_abs_gap,
        "median_ratio_scaled_over_scale1": ratio,
        "expected_ratio_if_applied": K**-0.5,
    }

    # --- Convention 2: does fla compute in float32? ----------------------------------------
    out["fla_output_dtype_for_float64_input"] = str(o_scale1.dtype)
    o_from_f32_inputs, _ = naive_recurrent_gated_delta_product(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g_head.float(),
        beta=beta.float(),
        scale=1.0,
        cu_seqlens=None,
        num_householder=R,
        output_final_state=True,
    )
    denom_f = o_scale1.double().abs().max().item()
    out["convention_2_float32_internal"] = {
        "f64_input_vs_f32_input_max_abs": (o_scale1.double() - o_from_f32_inputs.double())
        .abs()
        .max()
        .item(),
        "f64_input_vs_f32_input_relative": (o_scale1.double() - o_from_f32_inputs.double())
        .abs()
        .max()
        .item()
        / denom_f,
        "f32_input_output_dtype": str(o_from_f32_inputs.dtype),
    }

    # --- The anchor comparison itself, exactly as the test performs it ---------------------
    sys.path.insert(0, "/scratch/users/ericrcwu/agent-runs/dp2-kda-p0/OLMo-core/src")
    from olmo_core.nn.attention.kda_householder_torch import (  # type: ignore[import-not-found]
        kda_householder_torch,
    )

    o_mine, _ = kda_householder_torch(
        q, k, v, g_channel, beta, num_householder=R, scale=1.0, output_final_state=True
    )
    denom = o_scale1.double().abs().max().item()
    diff = (o_mine.double() - o_scale1.double()).abs().max().item()
    rel = diff / denom
    o_f32, _ = kda_householder_torch(
        q.float(),
        k.float(),
        v.float(),
        g_channel.float(),
        beta.float(),
        num_householder=R,
        scale=1.0,
        output_final_state=True,
    )
    float32_floor = (o_f32.double() - o_mine.double()).abs().max().item() / denom
    budget = max(10.0 * float32_floor, 1e-6)
    out["anchor_comparison"] = {
        "max_abs_diff": diff,
        "relative": rel,
        "float32_floor": float32_floor,
        "budget": budget,
        "passes": bool(rel < budget),
    }

    # And what the WRONG convention would produce, for contrast.
    o_mine_scaled, _ = kda_householder_torch(
        q, k, v, g_channel, beta, num_householder=R, scale=K**-0.5, output_final_state=True
    )
    out["anchor_with_wrong_scale_convention"] = {
        "relative": (o_mine_scaled.double() - o_scale1.double()).abs().max().item() / denom
    }

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
