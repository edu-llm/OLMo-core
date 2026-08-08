"""Premise-exposure histogram for the P3 v3 training corpus.

Exposure of a premise = (# training rows whose fact block cites it) x EPOCHS.
Identity = premise name (the label shown in "I know these mathematical
statements:"); `local_inputs` typing lemmas are excluded, matching the
DATASET-DESIGN "fact uses" definition. Read-only over corpus-v3/shards.

Outputs:
  figures/premise-exposure-histogram.png
  figures/premise-exposure-stats.json
"""

import json
import os
from collections import Counter
from pathlib import Path

# Keep matplotlib's cache inside the workspace (sandbox-writable).
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".mplcache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(os.environ.get("P3_CORPUS_ROOT", Path(__file__).resolve().parents[1]))
SHARDS = Path(os.environ.get("P3_CORPUS_SHARDS", ROOT / "corpus-v3" / "shards"))
OUTDIR = Path(os.environ.get("P3_FIGURES_DIR", ROOT / "figures"))
EPOCHS = 13
FAMILIES = ["enigma", "isabelle", "metamath", "mizar", "prf2", "thproofs"]
THRESHOLDS = [80, 100, 200, 500]


def count_uses():
    uses = Counter()
    rows = 0
    for fam in FAMILIES:
        with open(os.path.join(SHARDS, f"{fam}.jsonl")) as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("heldout", 0) == 1:
                    continue
                rows += 1
                for name in (row.get("facts") or {}):
                    uses[name] += 1
    return uses, rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    uses, rows = count_uses()

    exposures = np.array([c * EPOCHS for c in uses.values()], dtype=np.int64)
    total_uses = int(exposures.sum() // EPOCHS)
    n_uniq = exposures.size

    stats = {
        "epochs": EPOCHS,
        "train_rows": rows,
        "unique_premises": n_uniq,
        "total_premise_uses": total_uses,
        "median_exposures": int(np.median(exposures)),
        "mean_exposures": float(exposures.mean()),
        "max_exposures": int(exposures.max()),
        "thresholds": {},
    }
    for thr in THRESHOLDS:
        sel = exposures[exposures >= thr]
        stats["thresholds"][thr] = {
            "premises": int(sel.size),
            "share_premises": sel.size / n_uniq,
            "share_uses": int(sel.sum()) / int(exposures.sum()),
        }

    with open(os.path.join(OUTDIR, "premise-exposure-stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))

    # ---- figure: histogram (left) + cumulative share-of-uses (right) ----
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    lo, hi = 13, int(exposures.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 44)
    counts, edges = np.histogram(exposures, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    colors = ["#c44e52" if c < 100 else "#4c72b0" for c in centers]
    axL.bar(edges[:-1], counts, width=widths, align="edge",
            color=colors, edgecolor="white", linewidth=0.3)
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.axvline(100, color="black", ls="--", lw=1.5)
    axL.text(100, counts.max() * 1.1, "  100 exposures\n  (memorization threshold,\n  Allen-Zhu & Li 2024)",
             fontsize=9, va="top", ha="left")
    axL.axvline(stats["median_exposures"], color="#55a868", ls=":", lw=1.5)
    axL.text(stats["median_exposures"], counts.max() * 0.9,
             f"median {stats['median_exposures']}  ", fontsize=9, va="top", ha="right", color="#2d6a45")
    axL.set_xlabel("Exposures over training  (fact-uses \u00d7 13 epochs)")
    axL.set_ylabel("# unique premises")
    axL.set_title("Premise exposure distribution")
    axL.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#c44e52"),
        plt.Rectangle((0, 0), 1, 1, color="#4c72b0"),
    ], labels=["< 100 exposures (long tail)", "\u2265 100 exposures (saturated)"],
        loc="upper right", fontsize=9, framealpha=0.9)

    order = np.sort(exposures)[::-1]
    cum = np.cumsum(order) / order.sum()
    axR.plot(order, cum * 100, color="#4c72b0", lw=2)
    axR.set_xscale("log")
    axR.axvline(100, color="black", ls="--", lw=1.5)
    s100 = stats["thresholds"][100]["share_uses"] * 100
    axR.scatter([100], [s100], color="#c44e52", zorder=5)
    axR.annotate(f"\u2265100 exp:\n{stats['thresholds'][100]['share_premises']*100:.1f}% of premises\n"
                 f"= {s100:.1f}% of all uses",
                 xy=(100, s100), xytext=(130, s100 - 34),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="#c44e52"))
    axR.set_ylim(0, 102)
    axR.set_xlabel("Exposure threshold")
    axR.set_ylabel("% of all premise-uses at \u2265 threshold")
    axR.set_title("Coverage: few premises carry most uses")
    axR.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    axL.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle(
        f"P3 formal-proof corpus (v3): {n_uniq:,} unique premises, {total_uses:,} premise-uses, 13 epochs",
        fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "premise-exposure-histogram.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
