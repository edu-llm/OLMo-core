"""Judged pedagogy quality vs math retention, at the final checkpoint of every judged run.

This is the learning-forgetting trade-off with the REAL new-task metric on the x axis. Figure 3
uses held-out pedagogy NLL as a cheap per-checkpoint proxy; here the x axis is the blind
LLM-judge score (8-dim rubric, 40 held-out dialogues), which only exists for final checkpoints.

Both axes are measured WITHOUT a system instruction: gen_pedagogy.py strips the system message,
and the math probes carry no SI either. So these points describe cell C of the 2x2 (SFT, no SI),
not the deployment config (SFT + pedagogy SI).

Two judging batches are pooled and drawn distinctly, because the judge scores each batch
independently and they are NOT freely comparable — see the anchor spread annotated on the figure.

    python eval/plot_judge_vs_math.py --sweep out/ckpt_sweep_bare_hint250.jsonl
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# judge tag -> run name in the sweep JSONL. The judge's "impl2" candidate is the POC vanilla
# adapter (VANILLA_SFT=checkpoint-923 in impl3_h200.sbatch), which the sweep calls poc-c923.
TAG_TO_RUN = {"base": "base", "impl2": "poc-c923", "impl2-rerun": "impl2-rerun"}

# (judge_summary path, batch label, marker edge). Batch 1 judged 14 candidates together;
# the T=451 control was generated and judged later, in its own batch.
BATCHES = [
    ("eval/llm_judge/judge_summary.json", "main sweep batch", "k"),
    ("eval/llm_judge/t451/judge_summary.json", "T=451 control batch", "#7f2fbf"),
]


def disp(tag):
    return {"impl2": "SFT (POC)", "impl2-rerun": "SFT (rerun)", "base": "base"}.get(
        tag, tag.replace("impl3-", "")
    )


def collect(args):
    """[(tag, batch_idx, judge, math_hint, math_bare, variant, temperature)] for final ckpts."""
    rows = [json.loads(l) for l in open(args.sweep, encoding="utf-8") if l.strip()]
    final = {}
    for r in rows:
        run = r["run"]
        if run not in final or (r.get("step") or 0) > (final[run].get("step") or 0):
            final[run] = r

    pts, missing = [], []
    for bi, (path, _, _) in enumerate(BATCHES):
        if not os.path.exists(path):
            continue
        summary = json.load(open(path, encoding="utf-8"))
        for tag, v in summary.items():
            run = TAG_TO_RUN.get(tag, tag)
            row = final.get(run)
            if row is None:
                missing.append(tag)
                continue
            # A candidate judged in both batches is kept only from the first, so the anchors do
            # not double-plot; the batch-2 copies are what the spread annotation reports instead.
            if bi > 0 and any(p[0] == tag for p in pts):
                continue
            pts.append((tag, bi, v["OVERALL"], row.get("math_hint"), row.get("math_bare"),
                        row.get("variant"), row.get("temperature")))
    if missing:
        print(f"[warn] judged but absent from sweep: {', '.join(sorted(set(missing)))}")
    return pts


def place_labels(ax, labels):
    """Annotate points, picking a corner per label so they do not collide.

    Several runs land on identical accuracies (every collapsed variant-b config sits at math_hint
    0.212), so a fixed up-right offset stacks their names on top of each other. Work in normalised
    axis coordinates and greedily take, for each label, whichever of the four corners is furthest
    from the markers and from the labels already placed.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    norm = lambda x, y: ((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))
    pts = [norm(x, y) for x, y, *_ in labels]
    corners = [(0.030, 0.028), (0.030, -0.040), (-0.030, 0.028), (-0.030, -0.040)]

    placed = []
    for i, (x, y, text, color) in enumerate(labels):
        px, py = pts[i]
        best = None
        for dx, dy in corners:
            ax_, ay_ = px + dx, py + dy
            others = [p for j, p in enumerate(pts) if j != i] + placed
            d = min((((ax_ - ox) ** 2 + (ay_ - oy) ** 2) ** 0.5 for ox, oy in others),
                    default=9.9)
            # Nudge away from the frame: a label pushed outside the axes is worse than a near miss.
            if not (0.02 < ax_ < 0.98 and 0.02 < ay_ < 0.98):
                d -= 0.25
            if best is None or d > best[0]:
                best = (d, dx, dy)
        _, dx, dy = best
        placed.append((px + dx, py + dy))
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(9 if dx > 0 else -9, 7 if dy > 0 else -13),
                    ha="left" if dx > 0 else "right",
                    fontsize=10, fontweight="bold", zorder=10, color=color,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def anchor_spread():
    """How far apart the same vanilla-SFT recipe landed. This is the x-axis noise floor."""
    vals = {}
    for path, label, _ in BATCHES:
        if os.path.exists(path):
            s = json.load(open(path, encoding="utf-8"))
            for tag in ("impl2", "impl2-rerun", "base"):
                if tag in s:
                    vals.setdefault(tag, []).append(s[tag]["OVERALL"])
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="out/ckpt_sweep_bare_hint250.jsonl")
    p.add_argument("--out", default="out/figures/judge_vs_math.png")
    args = p.parse_args()

    pts = collect(args)
    if not pts:
        raise SystemExit("no judged runs matched the sweep")

    # Same encoding as figure 3: colour = temperature rank, shape = variant.
    temps = sorted({t for *_, t in pts if t})
    trank = {t: (i / (len(temps) - 1) if len(temps) > 1 else 0.5) for i, t in enumerate(temps)}
    turbo = plt.get_cmap("turbo")
    cmap_T = lambda f: turbo(0.06 + 0.88 * f)
    MARKER = {"a": "s", "b": "o"}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.4))

    for ax, ykey, ylabel, title in (
        (axes[0], 3, "Prior-task score: GSM8K accuracy (hinted)", "Hinted math"),
        (axes[1], 4, "Prior-task score: GSM8K accuracy (bare question)", "Bare math"),
    ):
        xs, ys, labels = [], [], []
        for tag, bi, judge, mh, mb, variant, T in pts:
            y = (mh, mb)[ykey - 3]
            if y is None:
                continue
            xs.append(judge)
            ys.append(y)
            edge = BATCHES[bi][2]
            if tag == "base":
                ax.scatter(judge, y, marker="*", s=520, color="black", zorder=8,
                           edgecolor=edge, linewidths=1.0)
            elif variant in MARKER and T:
                ax.scatter(judge, y, marker=MARKER[variant], s=150, color=cmap_T(trank[T]),
                           edgecolor=edge, linewidths=1.3, zorder=5)
            else:  # vanilla SFT anchors
                ax.scatter(judge, y, marker="X", s=210, color="black",
                           edgecolor=edge, linewidths=1.3, zorder=7)
            labels.append((judge, y, disp(tag),
                           "black" if tag in TAG_TO_RUN else cmap_T(trank[T])))
        place_labels(ax, labels)

        r = float(np.corrcoef(xs, ys)[0, 1])
        rs = float(np.corrcoef(np.argsort(np.argsort(xs)).astype(float),
                               np.argsort(np.argsort(ys)).astype(float))[0, 1])
        ax.set_xlabel("New-task quality: blind LLM-judge pedagogy score (0–1, no SI)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title}   —   Pearson r = {r:+.2f},  Spearman = {rs:+.2f}  (n={len(xs)})",
                     fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)

    # The x axis is only as sharp as the judge's own reproducibility: two runs of the same
    # vanilla-SFT recipe, judged in the same batch, are the widest calibration we have.
    spread = anchor_spread()
    note = []
    if len(spread.get("impl2", [])) > 1:
        note.append(f"same POC adapter judged twice: {spread['impl2'][0]:.3f} vs {spread['impl2'][1]:.3f}")
    if "impl2-rerun" in spread and "impl2" in spread:
        note.append(f"two vanilla-SFT runs in one batch: {spread['impl2'][-1]:.3f} vs {spread['impl2-rerun'][0]:.3f}")
    if len(spread.get("base", [])) > 1:
        note.append(f"base judged twice: {spread['base'][0]:.3f} vs {spread['base'][1]:.3f}")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="black", marker="*", ms=16, ls="none", label="base $\\pi_0$"),
        Line2D([], [], color="black", marker="X", ms=11, ls="none", label="vanilla SFT"),
        Line2D([], [], color="0.35", marker="s", ms=9, ls="none", label="$\\bf{square}$ = variant a"),
        Line2D([], [], color="0.35", marker="o", ms=9, ls="none", label="$\\bf{circle}$ = variant b"),
    ]
    handles += [Line2D([], [], color="w", marker="o", ms=10, ls="none", mec=e, mew=1.6, label=lab)
                for _, lab, e in BATCHES if os.path.exists(_)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.035))

    fig.suptitle("Judged pedagogy vs math retention at the final checkpoint of every judged run\n"
                 "both axes measured with NO system instruction (2x2 cell C)", fontsize=15)
    if note:
        fig.text(0.5, 0.015, "Judge noise floor — " + ";  ".join(note) +
                 ".  Horizontal differences smaller than this are not interpretable.",
                 ha="center", fontsize=10, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f'{"run":<16}{"judge":>8}{"math_hint":>11}{"math_bare":>11}  batch')
    print("-" * 52)
    for tag, bi, judge, mh, mb, _, _ in sorted(pts, key=lambda p: -p[2]):
        print(f"{disp(tag):<16}{judge:>8.3f}{mh:>11.3f}{mb:>11.3f}  {BATCHES[bi][1]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
