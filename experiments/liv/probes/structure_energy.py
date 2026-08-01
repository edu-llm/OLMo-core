"""How much activation-weighted energy does each CHEAP STRUCTURE retain, at matched cost?

WHY THIS EXISTS: the L40S benchmark killed P1's latency claim (fused r=128 + CUDA graphs is
still 8.2% SLOWER than dense) and showed `grouped` winning by +15%. But grouped g=4 and
lowrank r=128 cost EXACTLY the same -- 0.25x dense params, 10 MiB/token. So the real question
is no longer "factorize or not", it is "WHICH cheap structure", decided on quality.

We already know low-rank r=128 retains 92.6% of activation-weighted energy. We have no
comparable number for block-diagonal. This computes it on the released checkpoint, so the
quality question has a prior before we spend a single GPU-hour training.

METRIC: relative retained energy in the metric that governs output error,
    ||W_s . Sigma_x^{1/2}||_F^2  /  ||W . Sigma_x^{1/2}||_F^2
for each structure W_s at matched parameter count:
    lowrank r     -> optimal rank-r truncation of (W . Sigma^{1/2})  [Eckart-Young]
    grouped  g    -> zero all off-diagonal blocks of W, then weight
    (a random mask of the same density is included as the null)

TWO LIMITS, stated because they bound what this can conclude:
  1. This measures how well each structure APPROXIMATES TRAINED DENSE WEIGHTS. That is a
     proxy for from-scratch trainability, not the same thing -- GaLore's plain W=BA collapses
     from scratch (142.53 vs 15.56 ppl at 1B) at more generous rank fractions than these.
     A structure that approximates badly might still train fine, and vice versa.
  2. Block-diagonal retention depends on CHANNEL ORDERING; low-rank does not. A learned
     permutation could raise the grouped number substantially, so treat grouped's score as a
     LOWER bound. We report the random-permutation spread to size that sensitivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL = "LiquidAI/LFM2-350M"
SEQ, TOKEN_TARGET = 2048, 32768
CALIB = Path("/tmp/moby.txt")
OUT = Path(__file__).with_name("structure_energy_results.json")


def retained_lowrank(A: torch.Tensor, r: int) -> float:
    """Optimal rank-r retention (Eckart-Young): sum of top-r squared singular values."""
    sv = torch.linalg.svdvals(A.double())
    e = sv**2
    return float(e[:r].sum() / e.sum())


def retained_grouped(W: torch.Tensor, S_half: torch.Tensor, g: int,
                     perm: torch.Tensor | None = None) -> float:
    """Retention of a block-diagonal mask on W (mask applied to W, THEN weighted by Sigma).

    Permutation handling: reordering channels means the mask moves, not the weights. Applying
    a block mask to (P W P^T) and weighting by (P Sigma^{1/2} P^T) is equivalent to applying
    the inverse-permuted mask to W and weighting by the original Sigma^{1/2}, because the
    Frobenius norm is invariant under permutation. Permuting W while leaving S_half in the
    original channel basis -- as an earlier version of this function did -- silently compares
    mismatched bases and yields a meaningless number.
    """
    d = W.shape[0]
    bs = d // g
    mask = torch.zeros(d, d, dtype=torch.bool)
    for i in range(g):
        mask[i * bs:(i + 1) * bs, i * bs:(i + 1) * bs] = True
    if perm is not None:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(d)
        mask = mask[inv][:, inv]
    full = (W.double() @ S_half).norm() ** 2
    kept = ((W.double() * mask) @ S_half).norm() ** 2
    return float(kept / full)


def retained_random_mask(W: torch.Tensor, S_half: torch.Tensor, density: float,
                         gen: torch.Generator) -> float:
    """Null control: a random mask of the same density as the block-diagonal one."""
    m = (torch.rand(W.shape, generator=gen) < density)
    full = (W.double() @ S_half).norm() ** 2
    kept = ((W.double() * m) @ S_half).norm() ** 2
    return float(kept / full)


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    d = model.config.hidden_size
    liv = [(i, l.conv) for i, l in enumerate(model.model.layers)
           if getattr(l, "conv", None) is not None and hasattr(l.conv, "in_proj")]

    acc = {i: torch.zeros(d, d, dtype=torch.float64) for i, _ in liv}
    seen = {i: 0 for i, _ in liv}

    def mk(idx):
        def hook(_m, inputs, _o):
            x = inputs[0].detach().reshape(-1, d).double()
            acc[idx] += x.T @ x
            seen[idx] += x.shape[0]
        return hook

    handles = [op.in_proj.register_forward_hook(mk(i)) for i, op in liv]
    raw = CALIB.read_text(encoding="utf-8", errors="ignore")
    s = raw.find("CHAPTER 1")
    ids = tok(raw[s if s > 0 else 0:], return_tensors="pt").input_ids[0]
    with torch.no_grad():
        for c in range(-(-TOKEN_TARGET // SEQ)):
            chunk = ids[c * SEQ:(c + 1) * SEQ]
            if chunk.numel() < 64:
                break
            model(input_ids=chunk.unsqueeze(0))
    for h in handles:
        h.remove()
    print(f"calibrated on {seen[liv[0][0]]} tokens\n")

    gen = torch.Generator().manual_seed(0)
    rows = []
    for i, op in liv:
        sigma = acc[i] / seen[i]
        ev, evec = torch.linalg.eigh(sigma)
        S_half = (evec * ev.clamp_min(0).sqrt()) @ evec.T
        W = op.in_proj.weight.detach()
        B, C, _V = W.chunk(3, dim=0)

        entry = {"layer": i, "rank_sigma_x": int((ev > ev.max() * 1e-10).sum()), "gates": {}}
        for tag, M in (("B_pregate", B), ("C_postgate", C)):
            A = (M.double() @ S_half).float()
            rec = {
                # matched pairs: (r=128, g=4) and (r=256, g=2) are equal-cost
                "lowrank_128": retained_lowrank(A, 128),
                "grouped_4": retained_grouped(M, S_half, 4),
                "lowrank_256": retained_lowrank(A, 256),
                "grouped_2": retained_grouped(M, S_half, 2),
                "random_mask_25pct": retained_random_mask(M, S_half, 0.25, gen),
                # permutation sensitivity of the block-diagonal score
                "grouped_4_perm_spread": None,
            }
            perms = [retained_grouped(M, S_half, 4, torch.randperm(d, generator=gen))
                     for _ in range(3)]
            rec["grouped_4_perm_spread"] = [min(perms), max(perms)]
            entry["gates"][tag] = rec
        rows.append(entry)
        g = entry["gates"]["B_pregate"]
        print(f"  L{i:>2}  r128 {g['lowrank_128']:.3f}  g4 {g['grouped_4']:.3f}   "
              f"r256 {g['lowrank_256']:.3f}  g2 {g['grouped_2']:.3f}   "
              f"rand25 {g['random_mask_25pct']:.3f}")

    OUT.write_text(json.dumps({"model": MODEL, "rows": rows}, indent=2))

    def m(key: str) -> float:
        v = [r["gates"][t][key] for r in rows for t in ("B_pregate", "C_postgate")]
        return sum(v) / len(v)

    print("\n=== retained activation-weighted energy, mean over gates x layers ===")
    print(f"{'structure':<22}{'params':>9}{'retained':>10}")
    for lbl, key, cost in (("lowrank r=128", "lowrank_128", "0.25x"),
                           ("grouped g=4", "grouped_4", "0.25x"),
                           ("random mask 25%", "random_mask_25pct", "0.25x"),
                           ("lowrank r=256", "lowrank_256", "0.50x"),
                           ("grouped g=2", "grouped_2", "0.50x")):
        print(f"{lbl:<22}{cost:>9}{m(key):>10.3f}")

    lo = [r["gates"][t]["grouped_4_perm_spread"][0] for r in rows for t in ("B_pregate", "C_postgate")]
    hi = [r["gates"][t]["grouped_4_perm_spread"][1] for r in rows for t in ("B_pregate", "C_postgate")]
    print(f"\ngrouped g=4 under random channel permutation: "
          f"[{sum(lo)/len(lo):.3f}, {sum(hi)/len(hi):.3f}] "
          f"(identity ordering scores {m('grouped_4'):.3f})")

    print("\n=== VERDICT at matched 0.25x cost ===")
    r128, g4 = m("lowrank_128"), m("grouped_4")
    print(f"  lowrank r=128 {r128:.3f}  vs  grouped g=4 {g4:.3f}  -> "
          f"{'LOW-RANK' if r128 > g4 else 'GROUPED'} retains more "
          f"({abs(r128-g4)*100:.1f} pp)")
    print("  Grouped already wins latency by +15%. If it also retains comparable energy,")
    print("  P1's low-rank framing is dominated and `grouped` should be the headline arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
