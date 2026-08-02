#!/usr/bin/env python
"""Draw the RL's-Razor / KL-forgetting figures from a run's master_summary.json.

Generalizes the POC's ``curve_run/analysis/make_*`` scripts (which hardcoded POC
paths) to any run. Feed it a JSON list of per-checkpoint points:

    [ {"point": "base", "step": 0,  "kl_new": 0.0,  "acc": 0.42},
      {"point": "c1",    "step": 1,  "kl_new": 0.05, "acc": 0.41},
      {"point": "c16",   "step": 16, "kl_new": 0.31, "acc": 0.33}, ... ]

``kl_new`` comes from run_kl_curve.py (kl_new_SI) and ``acc`` from the graded math
retention (fraction 0-1). ``forget`` is derived as base_acc - acc if absent.
With log-spaced checkpoints the low-KL knee is densely sampled, so the fit is anchored
where the trajectory actually bends.

    python plot_kl_forgetting.py --summary out/impl3-a-T2/master_summary.json \
        --out_dir out/impl3-a-T2/figures

It also handles the *sweep* shape, where a point is a whole run's final checkpoint rather
than a step along one trajectory (no ``step`` field, ``point`` looks like "impl3-a-T4").
Then points are labelled and colored by variant instead of by a step colorbar. Pick which
metric to put on the y-axis with ``--acc_key`` — for the tutor sweep the interesting axis is
pedagogy, not math retention:

    python plot_kl_forgetting.py --summary out/master_summary.json \
        --acc_key ped_OVERALL --acc_label "Pedagogy quality (0-1)" --out_dir out/figures
"""
import argparse
import json
import os
import re


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, help="master_summary.json (list of points).")
    p.add_argument("--out_dir", default=None, help="Where to write figures (default: alongside summary).")
    p.add_argument("--acc_is_pct", action="store_true", help="acc values are 0-100 (default 0-1).")
    p.add_argument("--acc_key", default="acc",
                   help="Field for the y-axis metric (default 'acc'; e.g. 'math_acc', 'ped_OVERALL').")
    p.add_argument("--acc_label", default=None, help="Axis label for --acc_key (default derived from the key).")
    return p.parse_args()


def _variant_of(point):
    """'impl3-a-T4' -> 'a'. Reference points keep their own name so they stand out."""
    if point in ("base", "impl2"):
        return point
    m = re.search(r"-([ab])-T", point)
    return m.group(1) if m else "other"


def _linfit(x, y):
    import numpy as np

    m, b = np.polyfit(x, y, 1)
    yh = m * x + b
    r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    return m, b, r2


def main():
    args = parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(args.summary))
    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.summary)) + "/figures")
    os.makedirs(out_dir, exist_ok=True)

    def step_of(r):
        if "step" in r:
            return r["step"]
        p = r["point"]
        return 0 if p == "base" else int("".join(ch for ch in p if ch.isdigit()) or 0)

    key = args.acc_key
    rows = [r for r in rows if r.get("kl_new") is not None and r.get(key) is not None]
    if not rows:
        raise SystemExit(f"no points have both 'kl_new' and '{key}' — check --acc_key against the summary")

    # A sweep summary has no 'step' (each point is a different run, not a step along one run).
    # Digit-scraping "impl3-a-T32" would invent a step of 332, so switch to labelled mode instead.
    trajectory = any("step" in r for r in rows)
    rows = sorted(rows, key=step_of) if trajectory else sorted(rows, key=lambda r: r["kl_new"])

    kl = np.array([r["kl_new"] for r in rows], dtype=float)
    acc = np.array([r[key] for r in rows], dtype=float)
    if args.acc_is_pct:
        acc = acc / 100.0

    base_acc = next((r[key] for r in rows if r["point"] == "base"), acc.max())
    if args.acc_is_pct:
        base_acc = base_acc / 100.0
    forget = np.array([r.get("forget", base_acc - a) for r, a in zip(rows, acc)], dtype=float) * 100.0

    acc_label = args.acc_label or key.replace("_", " ")
    xlabel = r"New-task KL    $\mathrm{KL}(\pi_0\|\pi)$"
    variants = [_variant_of(r["point"]) for r in rows]
    vcolors = {"base": "#888888", "impl2": "#1f8a65", "a": "#2e79b5", "b": "#7b64b8", "other": "#c06028"}

    def draw(ax, yvals):
        """Scatter shared by both figures: step-colored for a trajectory, variant-colored for a sweep."""
        if trajectory:
            steps = np.array([step_of(r) for r in rows])
            sc = ax.scatter(kl, yvals, c=steps, cmap="viridis", s=170, zorder=3, edgecolor="k", linewidths=0.7)
            return sc
        for v in dict.fromkeys(variants):
            idx = [i for i, vv in enumerate(variants) if vv == v]
            ax.scatter(kl[idx], yvals[idx], c=vcolors[v], s=170, zorder=3, edgecolor="k", linewidths=0.7,
                       label={"a": "variant a (base-surprise)", "b": "variant b (forward-KL)"}.get(v, v))
        for x, y, r in zip(kl, yvals, rows):
            ax.annotate(r["point"].replace("impl3-", ""), (x, y), textcoords="offset points",
                        xytext=(9, -3), fontsize=8, color="0.25")
        return None

    # ---- Figure 1: forgetting vs new-task KL (the "middle graph") ----
    m, b, r2 = _linfit(kl, forget)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    sc = draw(ax, forget)
    xs = np.linspace(kl.min(), kl.max(), 100)
    ax.plot(xs, m * xs + b, "--", c="gray", lw=2.2, label=f"linear fit  $R^2$={r2:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Drop vs base in {acc_label} (pts)")
    ax.set_title(f"KL vs change in {acc_label}")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.25)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax); cb.set_label("training step")
    fig.tight_layout()
    f1 = os.path.join(out_dir, "fig_kl_forgetting.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight")

    # ---- Figure 2: the metric itself vs new-task KL ----
    fig2, ax2 = plt.subplots(figsize=(7.6, 6.0))
    sc2 = draw(ax2, acc * 100.0)
    if trajectory:
        order = np.argsort(kl)
        ax2.plot(kl[order], (acc * 100.0)[order], "-", c="gray", lw=1.2, alpha=0.6)
    else:
        for ref, style in (("impl2", "--"), ("base", ":")):
            r = next((r for r in rows if r["point"] == ref), None)
            if r:
                ax2.axhline(r[key] * 100.0, ls=style, c=vcolors[ref], lw=1.5, alpha=0.9, label=f"{ref} level")
        ax2.legend(loc="best", framealpha=0.9, fontsize=9)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(f"{acc_label} (%)")
    ax2.set_title(f"{acc_label} vs new-task KL")
    ax2.grid(alpha=0.25)
    if sc2 is not None:
        cb2 = fig2.colorbar(sc2, ax=ax2); cb2.set_label("training step")
    fig2.tight_layout()
    f2 = os.path.join(out_dir, "fig_acc_vs_kl.png")
    fig2.savefig(f2, dpi=150, bbox_inches="tight")

    print(f"points={len(rows)}  y={key}  mode={'trajectory' if trajectory else 'sweep'}  "
          f"linear R2(drop~kl)={r2:.3f}")
    print(f"saved {f1}\nsaved {f2}")


if __name__ == "__main__":
    main()
