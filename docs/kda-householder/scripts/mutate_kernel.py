"""Mutation testing on the REAL Triton kernel source (not a Python emulator).

Copies src/olmo_core/nn/attention/kda_householder.py to a scratch module, applies one textual
mutation to the *Triton* source, imports the mutated module, and asks whether the repo's own
acceptance criterion catches it. The criterion is the strongest one the repo actually uses:
per-gradient RELATIVE error < 2e-2 vs the gradcheck-validated torch backend, over the union of
the shape regimes the repo's tests cover.

A mutation is CAUGHT if any gradient's relative error >= 2e-2, or the kernel raises/produces NaN.
"""
import importlib.util, json, os, shutil, sys, traceback
import torch
import torch.nn.functional as F

SRC = os.environ["OLMO_SRC"]
KERNEL = os.path.join(SRC, "olmo_core/nn/attention/kda_householder.py")
WORK = os.environ["MUT_WORK"]
os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, SRC)

BASE = open(KERNEL).read()
DEV = "cuda"
NAMES = ["dq", "dk", "dv", "dg", "dbeta"]
TOL = 2e-2

# (id, description, old, new). Each must be a UNIQUE substring of the kernel source.
MUTATIONS = [
    # --- R-walk off-by-ones (2) ---
    ("M01_rewalk_le", "R-walk off-by-one: `if j < i_r` -> `j <= i_r`",
     "                if j < i_r:", "                if j <= i_r:"),
    ("M02_rewalk_reverse_order", "R-walk off-by-one: reverse index `R-1-i_rr` -> `R-1-i_rr-1` guarded",
     "            i_r = R - 1 - i_rr", "            i_r = max(R - 2 - i_rr, 0)"),
    # --- dg placement (1) ---
    ("M03_dg_after_decay", "store dg AFTER the decay hand-off (b_dh already multiplied by b_a)",
     "        tl.store(p_dg, tl.sum(b_dh * b_s0, 1), mask=mask_k)\n        b_dh = b_dh * b_a[:, None]",
     "        b_dh = b_dh * b_a[:, None]\n        tl.store(p_dg, tl.sum(b_dh * b_s0, 1), mask=mask_k)"),
    ("M04_dg_uses_pre_not_s0", "dg uses the PRE-decay state instead of S^(0)",
     "        tl.store(p_dg, tl.sum(b_dh * b_s0, 1), mask=mask_k)",
     "        tl.store(p_dg, tl.sum(b_dh * (b_s0 / tl.exp(b_g)[:, None]), 1), mask=mask_k)"),
    # --- dropped dk term (1) ---
    ("M05_dk_drop_term2", "drop the second dk term (k inside w = k @ S^(r))",
     "            b_dk = tl.sum(b_dh * b_u[None, :], 1) + tl.sum(b_inner * b_dw[None, :], 1)",
     "            b_dk = tl.sum(b_dh * b_u[None, :], 1)"),
    # --- last = T / T-2 (2) ---
    ("M06_last_eq_T", "`last = T - 1` -> `last = T` (pass-2 starts one token past the end)",
     "    last = T - 1", "    last = T"),
    ("M07_last_eq_T_minus_2", "`last = T - 1` -> `last = T - 2`",
     "    last = T - 1", "    last = T - 2"),
    # --- four wrong decrements (4) ---
    ("M08_dec_pk_wrong", "pass-2 decrement: `p_k -= R*H*K` -> `p_k -= H*K` (missing R)",
     "        p_k -= R * H * K\n        p_v -= R * H * V", "        p_k -= H * K\n        p_v -= R * H * V"),
    ("M09_dec_pv_wrong", "pass-2 decrement: `p_v -= R*H*V` -> `p_v -= H*V` (missing R)",
     "        p_v -= R * H * V\n        p_beta -= R * H", "        p_v -= H * V\n        p_beta -= R * H"),
    ("M10_dec_pbeta_wrong", "pass-2 decrement: `p_beta -= R*H` -> `p_beta -= H` (missing R)",
     "        p_beta -= R * H\n        p_hs -= H * K * V", "        p_beta -= H\n        p_hs -= H * K * V"),
    ("M11_dec_phs_wrong", "pass-2 decrement: `p_hs -= H*K*V` -> `p_hs -= K*V` (missing H)",
     "        p_hs -= H * K * V", "        p_hs -= K * V"),
    ("M12_dec_pdk_wrong", "pass-2 decrement: `p_dk -= R*H*K` -> `p_dk -= H*K` (missing R)",
     "        p_dk -= R * H * K\n        p_dv -= R * H * V", "        p_dk -= H * K\n        p_dv -= R * H * V"),
    ("M13_dec_pdv_wrong", "pass-2 decrement: `p_dv -= R*H*V` -> `p_dv -= H*V` (missing R)",
     "        p_dv -= R * H * V\n        p_db -= R * H", "        p_dv -= H * V\n        p_db -= R * H"),
    # --- dropped masks (3) ---
    ("M14_drop_mask_dk_store", "drop mask on the dk store",
     "            tl.store(p_dk + i_r * H * K, b_dk, mask=mask_k)",
     "            tl.store(p_dk + i_r * H * K, b_dk)"),
    ("M15_drop_mask_dq_store", "drop mask on the dq store",
     "        tl.store(p_dq, b_dq, mask=mask_k)", "        tl.store(p_dq, b_dq)"),
    ("M16_drop_mask_hs_load", "drop mask on the pass-2 hs load",
     "        b_s0 = tl.load(p_hs, mask=mask_h, other=0.0).to(tl.float32) * b_a[:, None]",
     "        b_s0 = tl.load(p_hs).to(tl.float32) * b_a[:, None]"),
    ("M17_drop_mask_dg_store", "drop mask on the dg store",
     "        tl.store(p_dg, tl.sum(b_dh * b_s0, 1), mask=mask_k)",
     "        tl.store(p_dg, tl.sum(b_dh * b_s0, 1))"),
    # --- dbeta missing its i_v slot (1) ---
    ("M18_dbeta_no_iv_slot", "dbeta pointer drops its per-i_v slot (partials collide)",
     "    p_db = db_p + i_v.to(tl.int64) * s_db + ((bos + last) * R) * H + i_h",
     "    p_db = db_p + ((bos + last) * R) * H + i_h"),
    # --- dropped scale (1) ---
    ("M19_drop_scale_dq", "drop `scale` from dq",
     "        b_dq = scale * tl.sum(b_sr * b_do[None, :], 1)",
     "        b_dq = tl.sum(b_sr * b_do[None, :], 1)"),
    ("M20_drop_scale_dh", "drop `scale` from the dh accumulation",
     "        b_dh += (b_q * scale)[:, None] * b_do[None, :]",
     "        b_dh += b_q[:, None] * b_do[None, :]"),
    # --- extra: dq_p / dg_p slot collisions ---
    ("M21_dq_no_iv_slot", "dq pointer drops its per-i_v slot",
     "    p_dq = dq_p + i_v.to(tl.int64) * s_dq + ((bos + last) * H + i_h) * K + o_k",
     "    p_dq = dq_p + ((bos + last) * H + i_h) * K + o_k"),
    ("M22_dk_no_iv_slot", "dk pointer drops its per-i_v slot",
     "    p_dk = dk_p + i_v.to(tl.int64) * s_dk + (((bos + last) * R) * H + i_h) * K + o_k",
     "    p_dk = dk_p + (((bos + last) * R) * H + i_h) * K + o_k"),
    ("M23_dbeta_uses_u", "dbeta contracts with u instead of the residual",
     "            tl.store(p_db + i_r * H, tl.sum(b_du * b_resid))",
     "            tl.store(p_db + i_r * H, tl.sum(b_du * b_u))"),
    ("M24_dv_no_beta", "dv drops the beta factor",
     "            tl.store(p_dv + i_r * H * V, (b_beta * b_du).to(p_dv.dtype.element_ty), mask=mask_v)",
     "            tl.store(p_dv + i_r * H * V, b_du.to(p_dv.dtype.element_ty), mask=mask_v)"),
    ("M25_dh_sign", "dh accumulation uses +b_du instead of b_dw (sign/scale error)",
     "            b_dh += b_k[:, None] * b_dw[None, :]", "            b_dh += b_k[:, None] * b_du[None, :]"),
    ("M26_no_debug_barrier", "remove tl.debug_barrier() between the hs write and read",
     "    tl.debug_barrier()", "    pass  # barrier removed"),
    ("M27_dq_uses_s0", "dq reads S^(0) instead of S^(R)",
     "        b_dq = scale * tl.sum(b_sr * b_do[None, :], 1)",
     "        b_dq = scale * tl.sum(b_s0 * b_do[None, :], 1)"),
]

# Regimes: cover the shape space the repo's own tests cover, incl. K=128 multiwarp, ragged, varlen.
REGIMES = [
    ("R1_K64",       dict(B=2, T=32, H=2, K=64, V=64, R=1)),
    ("R2_K64",       dict(B=2, T=32, H=2, K=64, V=64, R=2)),
    ("R3_K64",       dict(B=2, T=32, H=2, K=64, V=64, R=3)),
    ("R2_K128_mw",   dict(B=1, T=32, H=2, K=128, V=64, R=2)),
    ("R2_ragged_K48",dict(B=2, T=32, H=2, K=48, V=64, R=2)),
    ("R2_ragged_V48",dict(B=2, T=32, H=2, K=64, V=48, R=2)),
    ("R2_partialV60",dict(B=2, T=32, H=2, K=48, V=60, R=3)),
    ("R2_varlen",    dict(B=1, T=24, H=2, K=64, V=64, R=2, cu=[0, 7, 24])),
    ("R2_alog16",    dict(B=2, T=32, H=2, K=64, V=64, R=2, alog=16.0, neg=True)),
]


def mk(B, T, H, K, V, R, seed=0, alog=1.0, neg=False, cu=None):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    def rnd(*s): return torch.randn(*s, generator=gen, device=DEV, dtype=torch.float32)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_()
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_()
    v = rnd(B, T * R, H, V).to(torch.bfloat16).requires_grad_()
    beta = rnd(B, T * R, H).sigmoid()
    if neg: beta = beta * 2.0
    beta = beta.to(torch.bfloat16).requires_grad_()
    a = torch.rand(H, generator=gen, device=DEV) * (alog - 1.0) + 1.0 if alog > 1.0 else torch.ones(H, device=DEV)
    g = (-a.view(1, 1, H, 1) * F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_()
    do = rnd(B, T, H, V).to(torch.bfloat16)
    cus = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEV)
    return q, k, v, g, beta, do, cus


def load_module(src_text, tag):
    path = os.path.join(WORK, f"kh_{tag}.py")
    with open(path, "w") as f: f.write(src_text)
    spec = importlib.util.spec_from_file_location(f"kh_{tag}", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"kh_{tag}"] = m
    spec.loader.exec_module(m)
    return m


def grads(fn, cfg, backend):
    kw = {kk: vv for kk, vv in cfg.items() if kk not in ("cu", "alog", "neg")}
    q, k, v, g, beta, do, cus = mk(**kw, alog=cfg.get("alog", 1.0), neg=cfg.get("neg", False),
                                   cu=cfg.get("cu"), seed=hash(str(cfg)) % 10000)
    leaves = [q, k, v, g, beta]
    o, _ = fn(q, k, v, g, beta, num_householder=cfg["R"], backend=backend, cu_seqlens=cus)
    return [t.float() for t in torch.autograd.grad(o, leaves, grad_outputs=do)]


base_mod = load_module(BASE, "base")
# Reference: the gradcheck-validated torch backend, per regime.
REF = {}
for name, cfg in REGIMES:
    REF[name] = grads(base_mod.chunk_kda_householder, cfg, "torch")

# Sanity: the UNMUTATED kernel must pass every regime.
print("=" * 108)
print("SANITY: unmutated Triton kernel vs torch backend (relative error, tol 2e-2)")
print("=" * 108, flush=True)
base_ok = True
for name, cfg in REGIMES:
    got = grads(base_mod.chunk_kda_householder, cfg, "triton")
    rels = []
    for gg, rr in zip(got, REF[name]):
        sc = rr.abs().max().item()
        rels.append((gg - rr).abs().max().item() / sc if sc > 0 else float("nan"))
    ok = all(r < TOL for r in rels)
    base_ok &= ok
    print(f"  {name:16s} " + "  ".join(f"{n}={r:.2e}" for n, r in zip(NAMES, rels)) +
          f"   {'PASS' if ok else 'FAIL'}", flush=True)
print(f"  BASELINE: {'ALL PASS' if base_ok else 'BASELINE ITSELF FAILS'}\n", flush=True)

RESULTS = []
print("=" * 108)
print(f"MUTATION TESTING on the REAL TRITON KERNEL: {len(MUTATIONS)} mutations x {len(REGIMES)} regimes")
print("CAUGHT = some regime shows relative error >= 2e-2, or raises, or NaN")
print("=" * 108, flush=True)
for mid, desc, old, new in MUTATIONS:
    if BASE.count(old) != 1:
        print(f"  {mid:24s} SKIPPED-BAD-PATTERN (count={BASE.count(old)}) {desc}", flush=True)
        RESULTS.append(dict(id=mid, desc=desc, verdict="BAD_PATTERN", detail=f"count={BASE.count(old)}"))
        continue
    mutated = BASE.replace(old, new, 1)
    caught_by, evidence = [], {}
    try:
        mod = load_module(mutated, mid.lower())
    except Exception as e:
        print(f"  {mid:24s} CAUGHT (import/compile error: {type(e).__name__})", flush=True)
        RESULTS.append(dict(id=mid, desc=desc, verdict="CAUGHT", detail=f"import {type(e).__name__}"))
        continue
    for name, cfg in REGIMES:
        if caught_by:
            break  # early exit: one regime detecting the mutation is enough
        try:
            got = grads(mod.chunk_kda_householder, cfg, "triton")
        except Exception as e:
            caught_by.append(name); evidence[name] = f"raised {type(e).__name__}"
            continue
        worst, worst_n = 0.0, ""
        bad = False
        for n, gg, rr in zip(NAMES, got, REF[name]):
            if not torch.isfinite(gg).all():
                bad = True; worst, worst_n = float("inf"), n + "(nonfinite)"; break
            sc = rr.abs().max().item()
            r = (gg - rr).abs().max().item() / sc if sc > 0 else 0.0
            if r > worst: worst, worst_n = r, n
        if bad or worst >= TOL:
            caught_by.append(name)
        evidence[name] = f"max_rel={worst:.3e} on {worst_n}"
    verdict = "CAUGHT" if caught_by else "*** SURVIVED (silent) ***"
    print(f"  {mid:24s} {verdict:26s} caught_by={len(caught_by)}/{len(REGIMES)}  {desc[:60]}", flush=True)
    if not caught_by:
        for name in evidence: print(f"      {name:16s} {evidence[name]}", flush=True)
    RESULTS.append(dict(id=mid, desc=desc, verdict=verdict, caught_by=caught_by, evidence=evidence))
    del mod; sys.modules.pop(mid.lower(), None); torch.cuda.empty_cache()

n_real = [r for r in RESULTS if r["verdict"] != "BAD_PATTERN"]
n_caught = [r for r in n_real if r["verdict"] == "CAUGHT"]
print(f"\nSUMMARY: {len(n_caught)}/{len(n_real)} mutations CAUGHT "
      f"({len(RESULTS) - len(n_real)} skipped as bad patterns)", flush=True)
for r in n_real:
    if r["verdict"] != "CAUGHT":
        print(f"  SURVIVED: {r['id']} - {r['desc']}", flush=True)

with open(os.environ["OUT_JSON"], "w") as f:
    json.dump(dict(baseline_ok=base_ok, results=RESULTS,
                   n_caught=len(n_caught), n_real=len(n_real)), f, indent=1)
print("WROTE", os.environ["OUT_JSON"], flush=True)
