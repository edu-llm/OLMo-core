"""Render the P7 2x2 pedagogy result as a 3D bar chart beside the per-dimension table.

Floor of the 3D panel is the 2x2 factor grid (base vs SFT) x (no SI vs pedagogy SI);
height is the overall LLM-judge pedagogy score, the unweighted mean of 8 rubric
dimensions over 16 held-out problems. The right panel breaks that mean out by dimension.

Source: P7_Inference_Engineering/llm_judge/judge_summary.json
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d

DEFAULT_SUMMARY = os.path.expanduser(
    "~/Documents/MericXing/MIT/Intern/AlphaAI/P7_Inference_Engineering/llm_judge/judge_summary.json"
)

# (setup key, x index = model, y index = system instruction, panel label, colour)
CELLS = [
    ("A_raw_noSI", 0, 0, "A", "#c0392b"),
    ("C_sft_noSI", 1, 0, "C", "#e08a1e"),
    ("B_raw_SI", 0, 1, "B", "#2e86c1"),
    ("D_sft_SI", 1, 1, "D", "#1e8449"),
]
COL_ORDER = ["A_raw_noSI", "B_raw_SI", "C_sft_noSI", "D_sft_SI"]
COL_HEAD = ["A\nbase\nno SI", "B\nbase\n+SI", "C\nSFT\nno SI", "D\nSFT\n+SI"]
COL_COLOUR = {k: c for k, _, _, _, c in CELLS}

XT = ["base\n(OLMo-2-1B-Instruct)", "SFT\n(LoRA, ckpt-923)"]
YT = ["no system\ninstruction", "pedagogy\nsystem instruction"]

# Short label, full key, provenance. Order matches aggregate.py.
DIMS = [
    ("NoReveal", "Revealing_of_the_Answer", "MRB"),
    ("Guidance", "Providing_Guidance", "MRB"),
    ("Action", "Actionability", "MRB"),
    ("Coher", "Coherence", "MRB"),
    ("Tone", "Tutor_Tone", "MRB"),
    ("Human", "Humanlikeness", "MRB"),
    ("StepLvl", "Step_Level_Guidance", "P7"),
    ("LoadFmt", "Load_Aware_Formatting", "P7"),
]


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def draw_3d(ax, scores):
    dx = dy = 0.55
    for key, xi, yi, label, colour in CELLS:
        z = scores[key]
        ax.bar3d(
            xi - dx / 2,
            yi - dy / 2,
            0.0,
            dx,
            dy,
            z,
            color=colour,
            alpha=0.93,
            edgecolor="white",
            linewidth=1.1,
            shade=True,
            zsort="max",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(XT, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(YT, fontsize=9)
    ax.set_zlim(0, 1.0)
    ax.set_zticks(np.arange(0, 1.01, 0.2))
    ax.set_zlabel("overall pedagogy score (0–1)", fontsize=10, labelpad=6)
    ax.tick_params(axis="z", labelsize=9)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.15, 1.15, 0.9))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d8d8d8")
    ax.grid(True, color="#e8e8e8", linewidth=0.6)


def label_3d(fig, ax, scores):
    """Bar labels as 2D overlays.

    mplot3d depth-sorts text against the bars, so a tall back bar hides a short
    front bar's label if the label is placed with ax.text. Project the bar top to
    display space and annotate there instead.
    """
    fig.canvas.draw()
    for key, xi, yi, label, colour in CELLS:
        z = scores[key]
        x2, y2, _ = proj3d.proj_transform(xi, yi, z, ax.get_proj())
        ax.annotate(
            f"{label}   {z:.3f}",
            xy=(x2, y2),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=colour,
            zorder=20,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=colour, lw=1.0, alpha=0.95),
        )


def draw_table(ax, summary):
    ax.axis("off")

    rows = []
    for short, key, src in DIMS:
        rows.append([f"{short}"] + [f"{summary[c][key]:.3f}" for c in COL_ORDER])
    rows.append(["OVERALL"] + [f"{summary[c]['OVERALL']:.3f}" for c in COL_ORDER])

    table = ax.table(
        cellText=rows,
        colLabels=["dimension"] + COL_HEAD,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    n_rows = len(rows)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d5d5")
        cell.set_linewidth(0.7)
        if r == 0:  # header
            cell.set_height(0.115)
            cell.set_text_props(fontweight="bold", fontsize=9.5)
            cell.set_facecolor("#f2f2f2")
            if c >= 1:
                cell.set_text_props(
                    fontweight="bold", fontsize=9.5, color=COL_COLOUR[COL_ORDER[c - 1]]
                )
        else:
            cell.set_height(0.085)
            short, key, src = DIMS[r - 1] if r - 1 < len(DIMS) else (None, None, None)
            if r == n_rows:  # OVERALL
                cell.set_facecolor("#eaeaea")
                cell.set_text_props(fontweight="bold")
            elif src == "P7":
                cell.set_facecolor("#eaf3fb")
            if c == 0:
                cell.set_text_props(
                    ha="left",
                    fontweight="bold" if r == n_rows else "normal",
                )
                cell.PAD = 0.06

    ax.set_title(
        "Per-dimension breakdown (0–1, higher is better)",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--out", default="out/figures/pedagogy_2x2_3d.png")
    p.add_argument(
        "--title", default="Pedagogy quality: fine-tuning × system instruction"
    )
    args = p.parse_args()

    summary = load_summary(args.summary)
    scores = {k: summary[k]["OVERALL"] for k in summary}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(15.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.06)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    axtb = fig.add_subplot(gs[0, 1])

    draw_3d(ax3d, scores)
    draw_table(axtb, summary)
    label_3d(fig, ax3d, scores)

    fig.suptitle(args.title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.015,
        "Blinded LLM judge, 8 rubric dimensions (6 MRBench verbatim + the two P7 additions, shaded), "
        "16 held-out problems, n=16 per cell.\nThe system instruction is worth +0.324 and fine-tuning "
        "+0.137; the two are near-additive (interaction −0.008).",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )

    fig.savefig(args.out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for key, _, _, label, _ in CELLS:
        print(f"  {label}  {key:<12} {scores[key]:.4f}")
    si = (scores["B_raw_SI"] + scores["D_sft_SI"]) / 2 - (
        scores["A_raw_noSI"] + scores["C_sft_noSI"]
    ) / 2
    sft = (scores["C_sft_noSI"] + scores["D_sft_SI"]) / 2 - (
        scores["A_raw_noSI"] + scores["B_raw_SI"]
    ) / 2
    inter = scores["D_sft_SI"] - scores["C_sft_noSI"] - scores["B_raw_SI"] + scores["A_raw_noSI"]
    print(f"\n  SI main effect   {si:+.4f}")
    print(f"  SFT main effect  {sft:+.4f}")
    print(f"  interaction      {inter:+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
