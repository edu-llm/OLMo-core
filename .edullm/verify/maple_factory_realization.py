"""L1 verification: does the Maple factory REALIZE the layout it serializes?

A config that serializes correctly but realizes the wrong per-layer pattern is the silent
failure L1 exists to prevent, so every assertion here reads the BUILT module -- never the
config -- and asserts a magnitude or an exact set, never mere existence.

Run on FarmShare's `gpu` partition (see contracts/farmshare-env.md). Emits `RESULT ` lines
incrementally so a crash still leaves a readable trail.
"""

import sys
import traceback

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

V = 100_352
FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str) -> None:
    global CHECKS
    CHECKS += 1
    status = "PASS" if ok else "FAIL"
    print(f"RESULT {status} {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    print(f"RESULT env torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"RESULT env device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"RESULT env olmo_core_from={TransformerConfig.__module__}", flush=True)
    import olmo_core

    print(f"RESULT env olmo_core_path={olmo_core.__file__}", flush=True)

    # The factory must exist under the names L7 templates against.
    for name in ("maple_scaled", "maple_r0", "maple_r1", "maple_r2", "maple_r3"):
        check(f"factory_exists/{name}", hasattr(TransformerConfig, name), f"hasattr -> {hasattr(TransformerConfig, name)}")

    # ---------------------------------------------------------------------------------
    # 1. The signature constraint: vocab_size must be the ONLY required argument, because
    #    the platform's dispatcher passes exactly that and nothing else.
    # ---------------------------------------------------------------------------------
    for name in ("maple_r0", "maple_r1", "maple_r2", "maple_r3"):
        try:
            factory = getattr(TransformerConfig, name)
            cfg = factory(vocab_size=V)  # EXACTLY how train_on_corpus.py calls it
            check(f"launchable/{name}", True, f"factory(vocab_size={V}) built, total={cfg.num_params:,}")
        except Exception as e:  # noqa: BLE001
            check(f"launchable/{name}", False, f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------------------------
    # 2. Param ledger against the ratified ladder, at all four rungs.
    # ---------------------------------------------------------------------------------
    # MEASURED on FarmShare job 1676547 (L40S) and independently reproduced bit-for-bit by
    # closed-form arithmetic. These replace the figures first ratified in the contract, which
    # were wrong in two places -- R0 was computed at 2.0x attention width, and R1-R3's "exactly
    # constant active" claim holds for active-minus-routers, not active. See STATUS.md.
    expected = {
        "R0": (208_939_520, 120_859_136),
        "R1": (841_773_056, 313_290_752),
        "R2": (1_446_539_264, 314_077_184),
        "R3": (2_656_071_680, 315_650_048),
    }
    actives = {}
    for rung, (exp_total, exp_active) in expected.items():
        # Build with the ledger assertion suppressed, so ONE wrong constant does not abort the
        # whole harness and cost another GPU slot to learn the rest.
        cfg = TransformerConfig._maple_config(
            vocab_size=V,
            d_model=(s := TransformerConfig.MAPLE_RUNGS[rung])["d_model"],
            n_layers=s["n_layers"],
            n_heads=s["n_heads"],
            n_kv_heads=s["n_kv_heads"],
            head_dim=128,
            num_experts=s["num_experts"],
            top_k=8,
            expert_hidden_size=s["d_model"] // 4,
        )
        total, active = cfg.num_params, cfg.num_active_params
        # Router params are ACTIVE at every rung (every token traverses the full router) and
        # they scale with E, so `active` cannot be constant across an E-sweep. Report the
        # router-excluded figure alongside, which is the quantity that IS invariant.
        routers = s["n_layers"] * s["d_model"] * s["num_experts"]
        print(
            f"RESULT ledger_detail/{rung}: total={total} active={active} routers={routers} "
            f"active_minus_routers={active - routers}",
            flush=True,
        )
        actives[f"{rung}/minus_routers"] = active - routers
        dt = 100 * (total - exp_total) / exp_total
        da = 100 * (active - exp_active) / exp_active
        check(
            f"ledger/{rung}/total",
            abs(dt) <= 1.0,
            f"{total:,} vs contract {exp_total:,} ({dt:+.3f}%)",
        )
        check(
            f"ledger/{rung}/active",
            abs(da) <= 1.0,
            f"{active:,} vs contract {exp_active:,} ({da:+.3f}%)",
        )

    # The E-sweep's central property. NOTE: it holds for active params EXCLUDING routers, not
    # for active params. Every token traverses the whole router, so router params (L*d*E) are
    # active by definition and grow with E -- 0.79M at R1 to 3.15M at R3. Active-minus-routers
    # is the quantity that is exactly invariant, and it is the one the FLOPs argument needs.
    r1 = actives["R1/minus_routers"]
    r2 = actives["R2/minus_routers"]
    r3 = actives["R3/minus_routers"]
    check(
        "ledger/active_minus_routers_exactly_constant_R1_R3",
        r1 == r2 == r3,
        f"R1={r1:,} R2={r2:,} R3={r3:,} (spread {max(r1, r2, r3) - min(r1, r2, r3)})",
    )

    # ---------------------------------------------------------------------------------
    # 3. THE CHECK THIS LANE EXISTS FOR: realized per-layer window sizes and rope.
    #    Build the model and interrogate the modules, not the config.
    # ---------------------------------------------------------------------------------
    for rung, n_layers, expect_globals in (("R1", 12, {3, 7, 11}), ("R0", 8, {3, 7})):
        cfg = TransformerConfig.maple_scaled(V, rung=rung)
        model = cfg.build(init_device="meta")

        windows: dict[int, object] = {}
        ropes: dict[int, bool] = {}
        for idx_str, block in model.blocks.items():
            i = int(idx_str)
            attn = block.attention
            if not isinstance(attn, Attention):
                check(f"realized/{rung}/attn_type", False, f"block {i} mixer is {type(attn).__name__}")
                continue
            windows[i] = attn.window_size
            ropes[i] = attn.rope is not None

        realized_globals = {i for i, w in windows.items() if w is None}
        realized_swa = {i: w for i, w in windows.items() if w is not None}

        check(
            f"realized/{rung}/global_layers",
            realized_globals == expect_globals,
            f"globals at {sorted(realized_globals)}, expected {sorted(expect_globals)} "
            f"(all windows: {[windows[i] for i in range(n_layers)]})",
        )
        check(
            f"realized/{rung}/swa_window_512",
            all(w == 512 for w in realized_swa.values()),
            f"sliding layers {sorted(realized_swa)} -> windows {sorted(set(realized_swa.values()))}",
        )
        # NoPE on globals: rope must be ABSENT exactly on the global layers.
        nope_layers = {i for i, has in ropes.items() if not has}
        check(
            f"realized/{rung}/nope_on_globals",
            nope_layers == expect_globals,
            f"rope absent at {sorted(nope_layers)}, expected {sorted(expect_globals)}",
        )

        # Partial rotary and theta, read off a built RoPE on a sliding layer.
        sliding_idx = sorted(realized_swa)[0]
        rope = model.blocks[str(sliding_idx)].attention.rope
        check(
            f"realized/{rung}/rotary_dim",
            rope.rotary_dim == 64,
            f"block {sliding_idx} rotary_dim={rope.rotary_dim}, expected 64 (=128*0.5)",
        )
        check(
            f"realized/{rung}/rope_theta",
            rope.theta == 10_000,
            f"block {sliding_idx} theta={rope.theta}, expected 10000 (tree default is 500000)",
        )

        # Attention width equals the residual stream, as built.
        attn0 = model.blocks["0"].attention
        d = cfg.d_model
        check(
            f"realized/{rung}/attn_width_1x",
            attn0.w_q.out_features == d and attn0.w_out.in_features == d,
            f"w_q.out={attn0.w_q.out_features} w_out.in={attn0.w_out.in_features} d_model={d}",
        )
        # Per-head QK-norm: the norm must be sized head_dim, not n_heads*head_dim.
        check(
            f"realized/{rung}/qk_norm_per_head",
            attn0.use_head_qk_norm and attn0.q_norm.weight.numel() == attn0.head_dim,
            f"use_head_qk_norm={attn0.use_head_qk_norm} "
            f"q_norm.numel={attn0.q_norm.weight.numel()} head_dim={attn0.head_dim}",
        )
        # Router and capacity, as built.
        moe = model.blocks["0"].feed_forward_moe
        check(
            f"realized/{rung}/top_k",
            moe.router.top_k == 8,
            f"router.top_k={moe.router.top_k}, expected 8 (stock default is 1)",
        )
        check(
            f"realized/{rung}/normalize_expert_weights",
            moe.router.normalize_expert_weights == 1.0,
            f"={moe.router.normalize_expert_weights!r}, expected 1.0",
        )
        check(
            f"realized/{rung}/no_expert_bias",
            moe.router.bias_gamma is None,
            f"bias_gamma={moe.router.bias_gamma!r}, expected None",
        )
        check(
            f"realized/{rung}/no_shared_expert",
            moe.shared_mlp is None,
            f"shared_mlp={type(moe.shared_mlp).__name__ if moe.shared_mlp else None}",
        )
        del model

    # ---------------------------------------------------------------------------------
    # 4. Gate mass == 1.0. Guards normalize_expert_weights, measured 0.161 unset.
    #    Needs real weights, so R0 on the GPU (or CPU if no card).
    # ---------------------------------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TransformerConfig.maple_scaled(V, rung="R0")
    model = cfg.build(init_device=dev)
    router = model.blocks["0"].feed_forward_moe.router
    torch.manual_seed(0)
    x = torch.randn(2, 128, cfg.d_model, device=dev, dtype=torch.float32)
    with torch.no_grad():
        # Call the REAL router forward rather than reimplementing its scoring. A
        # reimplementation can pass while the shipped path is broken, which is exactly the
        # failure mode this check is supposed to catch.
        expert_weights, _, _, _ = router(x)
        mass = expert_weights.float().sum(dim=-1)
    lo, hi = mass.min().item(), mass.max().item()
    check(
        "gate_mass/R0",
        0.999 <= lo and hi <= 1.001,
        f"per-token top-k weight sum in [{lo:.6f}, {hi:.6f}], required [0.999, 1.001] "
        f"(measured 0.161 when normalize_expert_weights is unset)",
    )

    # ---------------------------------------------------------------------------------
    # 5. The assertions must actually RAISE, not warn. A guard that does not fire is not a
    #    guard -- so prove each one fires by feeding it a violation.
    # ---------------------------------------------------------------------------------
    negatives = [
        ("mixed_geometry_2x", dict(rung="R1", n_heads=16)),
        ("bad_f_e", dict(rung="R1", expert_hidden_size=512)),
        ("bad_L_mod_4", dict(rung="R1", n_layers=10)),
        ("unknown_rung", dict(rung="R9")),
    ]
    for label, kw in negatives:
        try:
            TransformerConfig.maple_scaled(V, **kw)
            check(f"guard_raises/{label}", False, "built WITHOUT raising -- the guard is dead")
        except Exception as e:  # noqa: BLE001
            check(f"guard_raises/{label}", True, f"raised {type(e).__name__} as required")

    print(f"RESULT SUMMARY checks={CHECKS} failures={len(FAILURES)}", flush=True)
    for f in FAILURES:
        print(f"RESULT FAILURE {f}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("RESULT SUMMARY harness_crashed", flush=True)
        sys.exit(2)
