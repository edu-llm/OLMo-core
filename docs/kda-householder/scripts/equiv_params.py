"""Claim F: R=1 == KDA equivalence, and the param-count formula as a function of R.
Claim E: beta = 2*sigmoid reachability of beta==1; allow_neg_eigval=False behaviour.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.environ["OLMO_SRC"])
from olmo_core.nn.attention.recurrent import (
    KimiDeltaAttentionConfig,
    KimiDeltaHouseholderConfig,
)

OUT = {}

# ---------- F1: param count, config formula vs BUILT module, R = 1..4 ----------
print("=" * 104)
print("F1: num_params() formula vs actually-built module, and the R multiplier")
print("=" * 104, flush=True)
d_model, n_heads = 512, 8
rows = []
kda_cfg = KimiDeltaAttentionConfig(n_heads=n_heads)
kda_formula = kda_cfg.num_params(d_model)
kda_mod = kda_cfg.build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
kda_actual = sum(p.numel() for p in kda_mod.parameters())
print(
    f"  KimiDeltaAttention        formula={kda_formula:>10,d}  built={kda_actual:>10,d}  "
    f"match={kda_formula == kda_actual}",
    flush=True,
)
OUT["kda"] = dict(formula=kda_formula, built=kda_actual, match=kda_formula == kda_actual)
for R in (1, 2, 3, 4, 8):
    c = KimiDeltaHouseholderConfig(n_heads=n_heads, num_householder=R)
    f_ = c.num_params(d_model)
    m = c.build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    a_ = sum(p.numel() for p in m.parameters())
    rows.append(
        dict(
            R=R,
            formula=f_,
            built=a_,
            match=f_ == a_,
            ratio_vs_R1=None,
            ratio_vs_kda=f_ / kda_formula,
        )
    )
    print(
        f"  KimiDeltaHouseholder R={R} formula={f_:>10,d}  built={a_:>10,d}  match={f_ == a_}  "
        f"ratio_vs_KDA={f_ / kda_formula:.4f}",
        flush=True,
    )
    del m
r1 = rows[0]["formula"]
for r in rows:
    r["ratio_vs_R1"] = r["formula"] / r1
print(
    f"\n  R=1 == KimiDeltaAttention param count: {r1 == kda_formula}  ({r1:,d} vs {kda_formula:,d})",
    flush=True,
)
print(f"  R=4 / R=1 mixer-param multiplier: {rows[3]['formula'] / r1:.4f}x", flush=True)
OUT["hh_params"] = rows

# ---------- F1b: per-parameter diff R=1 vs R=4 (prove they are NOT aliases) ----------
print("\n" + "=" * 104)
print("F1b: per-parameter shapes, R=1 vs R=4 (are the configs really different?)")
print("=" * 104, flush=True)
m1 = KimiDeltaHouseholderConfig(n_heads=n_heads, num_householder=1).build(
    d_model, layer_idx=0, n_layers=1
)
m4 = KimiDeltaHouseholderConfig(n_heads=n_heads, num_householder=4).build(
    d_model, layer_idx=0, n_layers=1
)
d1 = {n: tuple(p.shape) for n, p in m1.named_parameters()}
d4 = {n: tuple(p.shape) for n, p in m4.named_parameters()}
diffs = {n: (d1[n], d4[n]) for n in d1 if d1[n] != d4[n]}
same = [n for n in d1 if d1[n] == d4[n]]
print(f"  parameters that CHANGE with R ({len(diffs)}):", flush=True)
for n, (a, b) in sorted(diffs.items()):
    print(f"    {n:28s} R1={a}  R4={b}", flush=True)
print(f"  parameters INVARIANT to R ({len(same)}): {sorted(same)}", flush=True)
OUT["shape_diff"] = {n: dict(R1=list(a), R4=list(b)) for n, (a, b) in diffs.items()}
OUT["shape_same"] = sorted(same)

# ---------- F1c: R=1 KimiDeltaHouseholder vs KimiDeltaAttention -- same NAMED params? ----------
kdan = {n: tuple(p.shape) for n, p in kda_mod.named_parameters()}
print(f"\n  KDA param names == HH(R=1) param names: {set(kdan) == set(d1)}", flush=True)
print(f"    only in KDA: {sorted(set(kdan) - set(d1))}", flush=True)
print(f"    only in HH1: {sorted(set(d1) - set(kdan))}", flush=True)
shp = {n: (kdan[n], d1[n]) for n in set(kdan) & set(d1) if kdan[n] != d1[n]}
print(f"    shared names with DIFFERENT shapes: {shp}", flush=True)
OUT["kda_vs_hh1_names"] = dict(
    equal=set(kdan) == set(d1),
    only_kda=sorted(set(kdan) - set(d1)),
    only_hh1=sorted(set(d1) - set(kdan)),
    shape_mismatch={k: [list(a), list(b)] for k, (a, b) in shp.items()},
)

# ---------- F2: R=1 computational equivalence to the KDA path, on GPU ----------
if torch.cuda.is_available():
    print("\n" + "=" * 104)
    print("F2: R=1 KimiDeltaHouseholder vs KimiDeltaAttention -- COMPUTATIONAL equivalence")
    print("      (same weights copied across; both fed identical input)")
    print("=" * 104, flush=True)
    torch.manual_seed(0)
    hh1 = (
        KimiDeltaHouseholderConfig(
            n_heads=n_heads,
            num_householder=1,
            dtype=__import__("olmo_core.config", fromlist=["DType"]).DType.bfloat16,
        )
        .build(d_model, layer_idx=0, n_layers=1)
        .cuda()
    )
    kda2 = (
        KimiDeltaAttentionConfig(
            n_heads=n_heads, dtype=__import__("olmo_core.config", fromlist=["DType"]).DType.bfloat16
        )
        .build(d_model, layer_idx=0, n_layers=1)
        .cuda()
    )
    sd_hh = hh1.state_dict()
    sd_kda = kda2.state_dict()
    shared = [n for n in sd_hh if n in sd_kda and sd_hh[n].shape == sd_kda[n].shape]
    missing = [n for n in sd_kda if n not in shared]
    kda2.load_state_dict({**sd_kda, **{n: sd_hh[n].clone() for n in shared}})
    print(f"  copied {len(shared)}/{len(sd_kda)} tensors; could NOT align: {missing}", flush=True)
    x = torch.randn(2, 64, d_model, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        o_hh = hh1(x)
        o_kda = kda2(x)
    d = (o_hh.float() - o_kda.float()).abs()
    sc = o_kda.float().abs().max().item()
    print(
        f"  max|diff| = {d.max().item():.3e}   rel = {d.max().item()/sc:.3e}   "
        f"bit-exact = {torch.equal(o_hh, o_kda)}",
        flush=True,
    )
    OUT["r1_equiv"] = dict(
        copied=len(shared),
        total=len(sd_kda),
        unaligned=missing,
        max_abs=d.max().item(),
        rel=d.max().item() / sc,
        bit_exact=bool(torch.equal(o_hh, o_kda)),
    )

# ---------- E: beta = 2*sigmoid, is beta==1 reachable? ----------
print("\n" + "=" * 104)
print("E: beta = 2*sigmoid(w_b(x)) -- is beta == 1 an interior, reachable point?")
print("=" * 104, flush=True)
torch.manual_seed(1)
hhn = KimiDeltaHouseholderConfig(n_heads=n_heads, num_householder=2, allow_neg_eigval=True).build(
    d_model, layer_idx=0, n_layers=1
)
hhn.init_weights(
    init_method=__import__(
        "olmo_core.nn.transformer.init", fromlist=["InitMethod"]
    ).InitMethod.normal,
    d_model=d_model,
    block_idx=0,
    num_blocks=1,
)
x = torch.randn(4, 256, d_model)
with torch.no_grad():
    pre = hhn.w_b(x)
    beta_neg = pre.sigmoid() * 2.0
    beta_pos = pre.sigmoid()
print(
    f"  allow_neg_eigval=True : beta range [{beta_neg.min():.6f}, {beta_neg.max():.6f}], "
    f"n within 1e-3 of 1.0 = {(beta_neg - 1.0).abs().lt(1e-3).sum().item()} / {beta_neg.numel()}",
    flush=True,
)
print(
    f"  allow_neg_eigval=False: beta range [{beta_pos.min():.6f}, {beta_pos.max():.6f}], "
    f"n within 1e-3 of 1.0 = {(beta_pos - 1.0).abs().lt(1e-3).sum().item()} / {beta_pos.numel()}",
    flush=True,
)
print(
    f"  code default for KimiDeltaHouseholderConfig.allow_neg_eigval = "
    f"{KimiDeltaHouseholderConfig().allow_neg_eigval}",
    flush=True,
)
# inversion singularity 1/(|k|^2 - 1/beta) with |k|=1 -> singular at beta=1
print(f"  inversion denominator |k|^2 - 1/beta at beta=1, |k|=1: {1.0 - 1.0/1.0}", flush=True)
OUT["beta"] = dict(
    neg_min=beta_neg.min().item(),
    neg_max=beta_neg.max().item(),
    n_near_1_neg=int((beta_neg - 1.0).abs().lt(1e-3).sum()),
    pos_min=beta_pos.min().item(),
    pos_max=beta_pos.max().item(),
    n_near_1_pos=int((beta_pos - 1.0).abs().lt(1e-3).sum()),
    default_allow_neg=KimiDeltaHouseholderConfig().allow_neg_eigval,
    numel=int(beta_neg.numel()),
)

# ---------- LM-scale param claim: 52.1M -> 71.2M, +37% ----------
print("\n" + "=" * 104)
print("F3: LM-scale claim -- 52.1M non-embed (R=1) -> 71.2M (R=4), +37%")
print("=" * 104, flush=True)
print(
    "  (mixer-only figures; the LM total also needs FFN + norms + lm_head, which the "
    "LM-side agent owns)",
    flush=True,
)
for d_m, nh, nl in [(512, 8, 12), (512, 8, 16), (512, 16, 12)]:
    p1 = KimiDeltaHouseholderConfig(n_heads=nh, num_householder=1).num_params(d_m)
    p4 = KimiDeltaHouseholderConfig(n_heads=nh, num_householder=4).num_params(d_m)
    print(
        f"  d_model={d_m} n_heads={nh} n_layers={nl}: mixer/layer R1={p1:,d} R4={p4:,d}  "
        f"x{p4/p1:.3f} ; all-layer delta = {(p4-p1)*nl/1e6:.2f}M",
        flush=True,
    )
    OUT.setdefault("lm_scale", []).append(
        dict(
            d_model=d_m,
            n_heads=nh,
            n_layers=nl,
            p1=p1,
            p4=p4,
            ratio=p4 / p1,
            delta_M=(p4 - p1) * nl / 1e6,
        )
    )

with open(os.environ["OUT_JSON"], "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print("\nWROTE", os.environ["OUT_JSON"], flush=True)
