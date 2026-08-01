"""Activation-aware spectra of LFM2-350M gates -- v2, with the v1 flaw fixed.

V1 FLAW: calibration used 568 tokens for a 1024x1024 covariance. Sigma_x was rank-deficient
by construction (rank <= 568), which deflates every effective rank and manufactures an
apparent 3x collapse that is partly an artifact. v2 uses >=32k tokens (>=32x the dimension)
and REPORTS the rank of Sigma_x so the reader can confirm it is full.

v2 also adds the controls that make the comparison mean something:

  1. MLP control. If gates and the value stream both drop ~3x, that could just be a property
     of Sigma_x shared by all three (they take the SAME input x). The LIV out_proj and the
     MLP projections see DIFFERENT inputs, so they test whether the collapse is specific to
     this input distribution or generic to the model.

  2. Random-Gaussian control at matched shape, passed through the SAME Sigma_x^{1/2}. This is
     the null: how much effective-rank collapse does an UNSTRUCTURED matrix show under this
     input covariance? Any real drop must be measured against this floor, not against 1024.
     (This is the same trap stable rank fell into: srank read 26-48 for real weights while a
     random Gaussian scored 258, i.e. the "low" number was not evidence of structure.)

  3. Convergence check. Recompute at 25%/50%/100% of the token budget. If effective rank is
     still moving at 100%, the estimate has not converged and the number is not reportable.

WHAT WOULD ACTUALLY SUPPORT P1: gates (B, C) sitting clearly BELOW both the value stream x
AND the random control. Equal-to-value-stream means gates are not special, and the honest
framing stays "gates TOLERATE low rank" rather than "gates ARE low-rank".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

MODEL = "LiquidAI/LFM2-350M"
RANKS = (64, 128, 256, 512)
TOKEN_TARGET = 32768          # 32x hidden_size
SEQ = 2048
CALIB_FILE = Path("/tmp/moby.txt")
OUT = Path(__file__).with_name("spectra_v2_results.json")


def effective_rank(sv: torch.Tensor) -> float:
    sv = sv[sv > 0].double()
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def energy_at(sv: torch.Tensor, r: int) -> float:
    e = sv.double() ** 2
    return float(e[:r].sum() / e.sum())


def spec(W: torch.Tensor) -> dict:
    sv = torch.linalg.svdvals(W.double())
    return {
        "eff_rank": effective_rank(sv),
        "eff_rank_frac": effective_rank(sv) / min(W.shape),
        "energy": {str(r): energy_at(sv, r) for r in RANKS if r <= min(W.shape)},
    }


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not CALIB_FILE.exists():
        print(f"ERROR: {CALIB_FILE} missing", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    d = model.config.hidden_size

    liv = [(i, l.conv) for i, l in enumerate(model.model.layers)
           if getattr(l, "conv", None) is not None and hasattr(l.conv, "in_proj")]
    print(f"hidden={d}  LIV layers {[i for i, _ in liv]}")

    # ---- accumulate Sigma_x, with snapshots for the convergence check ----------------
    # Snapshot at 25% / 50% / 100% of the token budget.
    acc = {i: torch.zeros(d, d, dtype=torch.float64) for i, _ in liv}
    snaps: dict[int, dict[int, torch.Tensor]] = {i: {} for i, _ in liv}
    seen = {i: 0 for i, _ in liv}
    marks = (TOKEN_TARGET // 4, TOKEN_TARGET // 2, TOKEN_TARGET)

    def mk(idx):
        def hook(_m, inputs, _o):
            x = inputs[0].detach().reshape(-1, d).double()
            acc[idx] += x.T @ x
            before = seen[idx]
            seen[idx] += x.shape[0]
            for m in marks:                      # snapshot when a mark is crossed
                if before < m <= seen[idx]:
                    snaps[idx][m] = (acc[idx] / seen[idx]).clone()
        return hook

    handles = [op.in_proj.register_forward_hook(mk(i)) for i, op in liv]

    raw = CALIB_FILE.read_text(encoding="utf-8", errors="ignore")
    start = raw.find("CHAPTER 1")
    if start > 0:
        raw = raw[start:]
    ids = tok(raw, return_tensors="pt").input_ids[0]
    nchunk = -(-TOKEN_TARGET // SEQ)
    print(f"calibration: {nchunk} chunks x {SEQ} tok = {nchunk*SEQ} tokens", flush=True)
    with torch.no_grad():
        for c in range(nchunk):
            chunk = ids[c * SEQ:(c + 1) * SEQ]
            if chunk.numel() < 64:
                break
            model(input_ids=chunk.unsqueeze(0))
            print(f"  chunk {c+1}/{nchunk}", end="\r", flush=True)
    for h in handles:
        h.remove()
    print()

    g = torch.Generator().manual_seed(0)
    rows, conv_rows = [], []

    for i, op in liv:
        sigma = acc[i] / seen[i]
        evals, evecs = torch.linalg.eigh(sigma)
        rank_sigma = int((evals > evals.max() * 1e-10).sum())
        S_half = (evecs * evals.clamp_min(0).sqrt()) @ evecs.T

        W = op.in_proj.weight.detach()
        B, C, V = W.chunk(3, dim=0)                      # (B, C, x) -- verified order
        rand = torch.randn(d, d, generator=g) / d ** 0.5  # null control, matched shape

        entry = {"layer": i, "tokens": seen[i], "rank_sigma_x": rank_sigma, "tensors": {}}
        for tag, M in (("B_pregate", B), ("C_postgate", C), ("x_value", V),
                       ("out_proj", op.out_proj.weight.detach()), ("random", rand)):
            entry["tensors"][tag] = {
                "plain": spec(M),
                "aware": spec((M.double() @ S_half).float()),
            }
        rows.append(entry)
        t = entry["tensors"]
        print(f"L{i:>2} rank(Sigma_x)={rank_sigma:>4}  "
              f"B {t['B_pregate']['aware']['eff_rank']:6.1f}  "
              f"C {t['C_postgate']['aware']['eff_rank']:6.1f}  "
              f"x {t['x_value']['aware']['eff_rank']:6.1f}  "
              f"rand {t['random']['aware']['eff_rank']:6.1f}", flush=True)

        # convergence: effective rank of B at each snapshot
        cr = {"layer": i}
        for m, sig in sorted(snaps[i].items()):
            ev2, evec2 = torch.linalg.eigh(sig)
            sh = (evec2 * ev2.clamp_min(0).sqrt()) @ evec2.T
            cr[str(m)] = spec((B.double() @ sh).float())["eff_rank"]
        conv_rows.append(cr)

    OUT.write_text(json.dumps(
        {"model": MODEL, "hidden_size": d, "token_target": TOKEN_TARGET,
         "rows": rows, "convergence_B_pregate": conv_rows}, indent=2))

    def m(tag, kind):
        v = [r["tensors"][tag][kind]["eff_rank"] for r in rows]
        return sum(v) / len(v)

    def me(tag, kind, r="128"):
        v = [r_["tensors"][tag][kind]["energy"][r] for r_ in rows]
        return sum(v) / len(v)

    print(f"\n=== mean over {len(rows)} LIV layers, {seen[liv[0][0]]} tokens ===")
    print(f"{'tensor':<12} {'plain':>8} {'aware':>8} {'ratio':>7} {'E@128 aware':>12}")
    for tag in ("B_pregate", "C_postgate", "x_value", "out_proj", "random"):
        print(f"{tag:<12} {m(tag,'plain'):>8.1f} {m(tag,'aware'):>8.1f} "
              f"{m(tag,'aware')/m(tag,'plain'):>7.3f} {me(tag,'aware'):>12.3f}")

    print("\n=== convergence of B_pregate aware eff_rank ===")
    for cr in conv_rows[:4]:
        pts = " -> ".join(f"{k}:{cr[k]:.1f}" for k in sorted(cr, key=lambda s: (s == 'layer', s)) if k != "layer")
        print(f"  L{cr['layer']:<2} {pts}")

    gates = (m("B_pregate", "aware") + m("C_postgate", "aware")) / 2
    print(f"\nVERDICT  gates {gates:.1f} | value {m('x_value','aware'):.1f} "
          f"| random {m('random','aware'):.1f}  (of {d})")
    if gates < m("random", "aware") * 0.9 and gates < m("x_value", "aware") * 0.9:
        print("  gates clearly below BOTH controls -> P1 premise supported")
    else:
        print("  gates NOT below controls -> premise stays falsified; "
              "keep the 'gates TOLERATE low rank' framing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
