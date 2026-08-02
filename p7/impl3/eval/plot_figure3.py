#!/usr/bin/env python
"""Figure 3 of RL's Razor, rebuilt from our sweep — one fit per (variant, temperature).

The paper's Figure 3 has three panels and treats every mid-training checkpoint as a point
(Appendix B.3: "Including mid-training checkpoints, this produced approximately 500 runs per
method"). We do the same, with each configuration playing the role the paper gives to a method:

    left    new-task performance   vs  prior-task score    learning-forgetting trade-off
    middle  new-task KL            vs  prior-task score    "KL predicts forgetting"  <- the claim
    right   new-task KL            vs  new-task performance  how far each config travels per unit gain

Input is the JSONL from sweep_ckpt_eval.py: one row per checkpoint with kl_new_SI, prior_score
and ped_nll. New-task performance is the held-out pedagogy NLL, so LOWER is better; the axis is
inverted on the panels that use it so "further right = learned more", matching the paper's
accuracy axes.

Each configuration gets its own fit over its own checkpoints. The middle panel additionally
carries a single pooled fit with its R^2 — that pooled fit is the paper's actual claim (all
methods collapsing onto one curve), so keeping both makes it visible whether our configurations
agree or separate.

    python eval/plot_figure3.py --data out/ckpt_sweep_eval.jsonl --out_dir out/figures
"""
import argparse
import json
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="out/ckpt_sweep_eval.jsonl")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fit", choices=("isotonic", "poly"), default="isotonic",
                   help="Trend per config: monotone isotonic (default) or a polynomial.")
    p.add_argument("--degree", type=int, default=2, help="Polynomial degree when --fit poly (2 = the paper's).")
    # Name the metric outright rather than going through 'prior_score'. The two are identical today
    # (all 194 rows), but 'prior_score' is an alias chosen by the sweep driver, so redefining it
    # there — as happened when IFEval was dropped from it — would silently change what this figure
    # plots. Hinted is the default because it is the condition the forgetting effect shows up in:
    # the boxing hint collides with the tutor persona, and SFT drops to 0.212 against base 0.664,
    # where bare only falls to 0.456. Bare is still worth plotting since it separates refusal from
    # genuine skill loss, but it understates the phenomenon.
    p.add_argument("--prior_key", default="math_hint", help="Prior-task metric (y of left/middle).")
    p.add_argument("--new_key", default="ped_nll", help="New-task metric (x of left, y of right).")
    # Default to the no-SI KL. The prior-task probes carry no system instruction, and variant a
    # learns an SI-GATED policy: it moves a lot with the SI present and barely at all without, so
    # measured with SI it sits far off variant b's curve despite forgetting about as little. The
    # no-SI KL is what collapses both families onto one curve (pooled R2 0.74 vs 0.37, 0.95 under
    # the monotone fit), and that method-invariance is the claim being tested. Note this is not a
    # uniform improvement: WITHIN variant b the with-SI KL predicts slightly better (0.84 vs 0.73).
    # Both remain available, and their gap is our measure of how gated a run is.
    p.add_argument("--kl_key", default="kl_ped_noSI")
    p.add_argument("--kl_alt", default="kl_new_SI",
                   help="Second KL condition, plotted against --kl_key in the robustness figure.")
    p.add_argument("--no_robustness", action="store_true",
                   help="Skip the KL-condition robustness figure.")
    p.add_argument("--suffix", default="",
                   help="Appended to the output filename, so alternative axes (e.g. --prior_key "
                        "math_bare) do not overwrite the default figure.")
    p.add_argument("--min_points", type=int, default=4, help="Configs with fewer checkpoints are not fit.")
    p.add_argument("--exclude", default="poc-c923",
                   help="Comma-separated runs to leave out of the figure. Defaults to the POC anchor, "
                        "which was measured only to check it matches impl2-rerun (it does, to within "
                        "1%% of the axis range) and now just double-plots the vanilla endpoint.")
    return p.parse_args()


def _pava(np, y):
    """Pool-adjacent-violators: the least-squares non-DECREASING fit to y in index order."""
    blocks = []  # [mean, count]
    for v in y:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    out = []
    for v, n in blocks:
        out.extend([v] * n)
    return np.array(out)


def disp(name):
    """Short label for a run. 'impl2-rerun' is our plain SFT baseline; the name is an artefact of
    it being a re-run of the Impl-2 recipe and means nothing to a reader of the figure."""
    return {"impl2-rerun": "SFT"}.get(name, name.replace("impl3-", ""))


def fit_curve(np, x, y, *, kind="isotonic", degree=1):
    """Fit a trend to (x, y) and return (xs, ys, r2) ready to plot, or None if underdetermined.

    Default is isotonic (monotone) regression rather than a straight line. A line imposes a
    constant rate of change these trajectories plainly do not have — the KL/NLL curves bend hard
    early and flatten later — so a linear fit understates the trend and its R^2 measures
    "how straight is this" rather than "is this monotone". Isotonic assumes only monotonicity,
    which is the actual claim being made about these curves, and infers the direction per
    configuration (variant a at low T moves the opposite way from everything else).
    """
    if len(x) < 3:
        return None
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]

    if kind == "isotonic":
        increasing = np.polyfit(xs, ys, 1)[0] >= 0 if len(set(xs)) > 1 else True
        pred = _pava(np, ys) if increasing else -_pava(np, -ys)
    else:
        if len(xs) < degree + 1 or len(set(xs)) < degree + 1:
            return None
        coef = np.polyfit(xs, ys, degree)
        xs = np.linspace(xs.min(), xs.max(), 160)
        pred = np.polyval(coef, xs)
        resid = y - np.polyval(coef, x)
        ss_tot = float(((y - np.mean(y)) ** 2).sum())
        return xs, pred, (1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"))

    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (*_smooth_monotone(np, xs, pred), r2)


def _smooth_monotone(np, xs, pred):
    """Render an isotonic fit as a smooth curve without breaking its monotonicity.

    Raw PAVA output is a staircase, which is visually noisy and implies discontinuities the
    underlying process does not have. PCHIP is shape-preserving: it is monotone on exactly the
    intervals where its input data are monotone, so smoothing here cannot introduce a wiggle that
    reverses the trend. Falls back to the staircase if SciPy is unavailable.
    """
    # Collapse each constant block of the staircase to a single knot at the block's x-centroid.
    # Interpolating through every point instead would trace the risers and keep the corners; the
    # block structure is the actual information PAVA extracted, and a curve through the centroids
    # carries it with far fewer knots.
    ux, uy, run_x = [], [], []
    for i, (x, y) in enumerate(zip(xs, pred)):
        if uy and y == uy[-1]:
            run_x.append(float(x))
        else:
            if run_x:
                ux.append(sum(run_x) / len(run_x))
            run_x = [float(x)]
            uy.append(float(y))
    if run_x:
        ux.append(sum(run_x) / len(run_x))

    if len(ux) == 1:  # PAVA found no trend at all: a flat line across the observed range
        return np.array([float(xs[0]), float(xs[-1])]), np.array([uy[0], uy[0]])
    if len(ux) < 3:
        return np.array(ux), np.array(uy)
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:
        return np.array(ux), np.array(uy)
    grid = np.linspace(ux[0], ux[-1], 300)
    return grid, PchipInterpolator(np.array(ux), np.array(uy))(grid)


def _spearman(np, a, b):
    """Rank correlation, so the comparison is about ORDER rather than the two KL scales."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def plot_kl_robustness(np, plt, pe, args, configs, names, colors, markers, base,
                       prior_label, out_dir):
    """Is the KL axis robust to whether the pedagogy SI is in context?

    The main figure measures new-task KL with the canonical pedagogy SI present
    (``kl_new_SI``); the same checkpoints are also scored without it (``kl_ped_noSI``).
    The two are not interchangeable — they differ by roughly 7x in scale, correlate at
    only r~0.7, and rank the two weighting variants in OPPOSITE orders. Variant a
    weights tokens by p_0^(1/T), so it barely perturbs the UNCONDITIONED distribution
    and scores well on a condition it hardly touches; the SI-conditioned number is the
    one that reflects the behaviour the model is actually being asked for. Publishing
    both is what keeps ``kl_new_SI`` from looking like a choice made after seeing the
    answer.

    Left panel is a slope graph over final checkpoints: every crossing is a pair of
    configurations the two conditions disagree about. Right panel re-draws the central
    "KL predicts forgetting" claim on the no-SI axis.
    """
    have = {n: [r for r in configs[n]
                if r.get(args.kl_key) is not None and r.get(args.kl_alt) is not None]
            for n in names}
    have = {n: v for n, v in have.items() if v}
    if len(have) < 3:
        print(f"skipping KL-condition robustness figure: <3 runs carry both "
              f"{args.kl_key} and {args.kl_alt}")
        return

    allsi = np.array([r[args.kl_key] for v in have.values() for r in v])
    allno = np.array([r[args.kl_alt] for v in have.values() for r in v])
    pear = float(np.corrcoef(allsi, allno)[0, 1])
    ratio = float(np.mean(allsi / np.maximum(allno, 1e-9)))

    order = sorted(have, key=lambda n: have[n][-1][args.kl_key])
    finals = {n: max(have[n], key=lambda r: r["step"]) for n in order}
    fsi = np.array([finals[n][args.kl_key] for n in order])
    fno = np.array([finals[n][args.kl_alt] for n in order])
    spear = _spearman(np, fsi, fno)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

    # --- left: slope graph, one line per config, crossings = disagreements -------------
    ax = axes[0]

    # Position by RANK, not by value. Several configs tie to within a thousandth of a nat
    # (impl2-rerun 0.761 vs b-T451 0.759), so plotting the values themselves stacks their
    # labels on top of each other — and the claim here is about order anyway, which rank
    # states directly and legibly.
    def ranks(v):
        return len(v) - 1 - np.argsort(np.argsort(v)).astype(float)  # lowest KL at the top

    rsi, rno = ranks(fsi), ranks(fno)
    for i, n in enumerate(order):
        c = colors[n]
        short = disp(n)
        ax.plot([0, 1], [rsi[i], rno[i]], "-", color=c, lw=2.0, alpha=0.9,
                marker=markers[n], ms=7, mec="k", mew=0.4)
        ax.annotate(f"{short}  {fsi[i]:.3f}", (0, rsi[i]), textcoords="offset points",
                    xytext=(-9, 0), ha="right", va="center", fontsize=8,
                    color=c, fontweight="bold")
        ax.annotate(f"{fno[i]:.3f}  {short}", (1, rno[i]), textcoords="offset points",
                    xytext=(9, 0), ha="left", va="center", fontsize=8,
                    color=c, fontweight="bold")
    ax.set_xlim(-0.66, 1.66)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"with SI\n({args.kl_key})", f"without SI\n({args.kl_alt})"], fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel("rank by final-checkpoint KL   (top = lowest KL = closest to base)")
    ax.set_title("Does the SI condition change the ranking?\n"
                 f"Spearman = {spear:+.2f} — every crossing is a pair the two conditions "
                 "disagree about", fontsize=11)

    # --- right: the central claim, re-drawn on the no-SI axis --------------------------
    ax = axes[1]
    for n in names:
        pts = [(r[args.kl_alt], r[args.prior_key]) for r in configs[n]
               if r.get(args.kl_alt) is not None and r.get(args.prior_key) is not None]
        if len(pts) < 2:
            continue
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        c = colors[n]
        ref = configs[n][0].get("variant") is None
        ax.scatter(x, y, color=c, s=70 if ref else 42, alpha=0.9, marker=markers[n],
                   edgecolor="k", linewidths=0.4, zorder=4 if ref else 3)
        j = int(np.argmax(x))
        ax.annotate(disp(n), (x[j], y[j]), textcoords="offset points",
                    xytext=(6, 4), fontsize=6.8, color="black" if ref else c,
                    fontweight="bold", zorder=10,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        fit = fit_curve(np, x, y, kind=args.fit, degree=args.degree) \
            if len(x) >= args.min_points else None
        if fit:
            fx, fy, _ = fit
            ax.plot(fx, fy, "-", color=c, lw=3.2 if ref else 2.0, alpha=0.95,
                    zorder=5 if ref else 2)
        else:
            ax.plot(x, y, "-", color=c, lw=1.0, alpha=0.5)
    if base is not None and base.get(args.prior_key) is not None:
        ax.scatter([0.0], [base[args.prior_key]], marker="*", s=380, color="black", zorder=7)
        ax.axhline(base[args.prior_key], ls=":", c="0.5", lw=1.2, alpha=0.8)
    ax.set_xlabel(r"New-task KL, no SI in context   $\mathrm{KL}(\pi_0\|\pi)$")
    ax.set_ylabel(prior_label)
    ax.set_title("KL vs forgetting, measured without the SI")
    ax.grid(alpha=0.25)

    fig.suptitle(f"Robustness of the KL axis to the SI condition — pooled r = {pear:.2f} "
                 f"over {len(allsi)} checkpoints, mean scale ratio {ratio:.1f}x", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"fig3_kl_condition_robustness{args.suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")

    print(f"\nKL condition robustness: pooled r={pear:.3f}, mean ratio {ratio:.2f}x, "
          f"Spearman over final checkpoints {spear:+.3f}")
    print(f'{"config":<14}{args.kl_key:>13}{args.kl_alt:>15}{"ratio":>8}')
    print("-" * 50)
    for i, n in enumerate(order):
        print(f"{disp(n):<14}{fsi[i]:>13.4f}{fno[i]:>15.4f}"
              f"{fsi[i] / max(fno[i], 1e-9):>8.2f}")
    print(f"saved {path}")


def main():
    args = parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.data)), "figures")
    os.makedirs(out_dir, exist_ok=True)

    base = next((r for r in rows if r["run"] == "base"), None)
    dropped = {s.strip() for s in args.exclude.split(",") if s.strip()}
    configs = defaultdict(list)
    for r in rows:
        if r["run"] == "base" or r.get(args.kl_key) is None or r["run"] in dropped:
            continue
        configs[r["run"]].append(r)
    if dropped:
        print(f"excluded from figure: {', '.join(sorted(dropped))}")
    for k in configs:
        configs[k].sort(key=lambda r: r["step"])
    if not configs:
        raise SystemExit(f"no per-checkpoint rows in {args.data}")

    names = sorted(configs, key=lambda n: (configs[n][0].get("variant") or "z",
                                           configs[n][0].get("temperature") or 0))

    # Colour encodes TEMPERATURE and shape encodes VARIANT, so the two are readable independently.
    # Cool = low T (heavily reweighted, far from vanilla) through warm = high T (-> vanilla SFT).
    # Both variants share one scale, which is the point: a-T8 and b-T8 get the same colour and
    # differ only in shape.
    #
    # Colour is assigned by the RANK of T among the swept values, not by T or log T. T=451 is the
    # deliberate "T -> infinity" limit check, and on any continuous scale it strands every other run
    # in one end of the palette: with log2, T=0.5..32 occupies 68% of the range and comes out as six
    # barely distinguishable steps. Ranking is still monotone in temperature, which is all the
    # encoding has to promise, and it spends the whole palette on the runs being compared.
    #
    # turbo: blue -> cyan -> green -> yellow -> orange -> red, i.e. cold/hot reads intuitively as
    # temperature while spending the entire spectrum rather than one half of it. Two other cues are
    # carried alongside for colour-vision deficiency, so no single hue judgement is load-bearing:
    # marker shape separates the variants, and every curve is labelled at its own endpoint.
    temps = sorted({t for n in names if (t := configs[n][0].get("temperature"))})
    trank = {t: (i / (len(temps) - 1) if len(temps) > 1 else 0.5) for i, t in enumerate(temps)}
    base_cmap = plt.get_cmap("turbo")
    # Trimmed at both ends: turbo terminates in near-black navy and near-black maroon, which read as
    # "dark" rather than as blue and red.
    cmap_T = lambda f: base_cmap(0.06 + 0.88 * f)
    MARKER = {"a": "s", "b": "o"}  # squares = variant a (base-surprise), circles = variant b (fwd-KL)
    colors, markers = {}, {}
    for n in names:
        v = configs[n][0].get("variant")
        T = configs[n][0].get("temperature")
        if v in MARKER and T:
            colors[n] = cmap_T(trank[T])
            markers[n] = MARKER[v]
        elif len(configs[n]) == 1:
            # Single-checkpoint anchors (e.g. the POC's checkpoint-923) have no trajectory and land
            # almost on the vanilla endpoint, so they need a shape nothing else uses.
            colors[n], markers[n] = "#d62728", "D"
        else:
            # vanilla SFT: the benchmark every Impl-3 config is judged against
            colors[n], markers[n] = "black", "X"

    prior_label = {
        "prior_score": "Prior-task score: GSM8K accuracy (hinted)",
        "math_hint": "Prior-task score: GSM8K accuracy (hinted)",
        "math_bare": "Prior-task score: GSM8K accuracy (bare question)",
        "math_hint_commit": "Answer-commit rate (hinted)",
        "math_hint_acc_given_commit": "GSM8K accuracy among attempts (hinted)",
        "math_acc": "Prior-task score: math accuracy",
    }.get(args.prior_key, f"Prior-task score: {args.prior_key}")

    # Name the KL condition on the axis. Both conditions get plotted and the two figures are
    # otherwise identical, but they say very different things: the prior-task probes carry no
    # system instruction, so only the no-SI KL is measured where the eval actually lives, and it
    # predicts forgetting far better (pooled R2 0.81 vs 0.37). An unlabelled axis makes the two
    # impossible to tell apart once a figure is pasted somewhere without its filename.
    kl_cond = {"kl_new_SI": "with SI in context",
               "kl_ped_noSI": "no SI in context"}.get(args.kl_key, args.kl_key)
    kl_label = r"New-task KL   $\mathrm{KL}(\pi_0\|\pi)$" + f"   [{kl_cond}]"

    panels = [
        ("new_vs_prior", args.new_key, args.prior_key,
         "New-task performance:  held-out pedagogy NLL", prior_label,
         "Learning vs forgetting", True),
        ("kl_vs_prior", args.kl_key, args.prior_key,
         kl_label, prior_label, "KL vs forgetting", False),
        ("kl_vs_new", args.kl_key, args.new_key,
         kl_label, "New-task performance:  pedagogy NLL", "KL vs new-task gain", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0))
    fit_report = []
    for ax, (pid, xk, yk, xlabel, ylabel, title, invert_x) in zip(axes, panels):
        allx, ally, panel_r2 = [], [], []
        for name in names:
            pts = [(r[xk], r[yk]) for r in configs[name] if r.get(xk) is not None and r.get(yk) is not None]
            if not pts:
                continue
            x = np.array([p[0] for p in pts], float)
            y = np.array([p[1] for p in pts], float)
            allx += list(x); ally += list(y)
            c = colors[name]
            ref = configs[name][0].get("variant") is None  # the vanilla-SFT reference run
            if len(x) == 1:
                # Drawn as a large hollow diamond on top of everything, and labelled in place: it
                # overlaps the vanilla endpoint to within 1% of the axis range, so nothing short of
                # a different shape and an arrow makes it findable.
                ax.scatter(x, y, facecolor="none", edgecolor=c, s=210, marker="D",
                           linewidths=2.4, zorder=9)
                ax.annotate(name, (x[0], y[0]), textcoords="offset points", xytext=(14, 14),
                            fontsize=8.5, color=c, fontweight="bold", zorder=9,
                            arrowprops=dict(arrowstyle="-", color=c, lw=1.0, alpha=0.8))
                continue
            ax.scatter(x, y, color=c, s=70 if ref else 42, alpha=0.9,
                       marker=markers[name], edgecolor="k", linewidths=0.4,
                       zorder=4 if ref else 3)
            # Label every curve where it ends. This is the accessibility fallback: colour and shape
            # both become optional, because the name is physically next to the line.
            j = int(np.argmax(x)) if not invert_x else int(np.argmin(x))
            ax.annotate(disp(name), (x[j], y[j]),
                        textcoords="offset points", xytext=(6, 4), fontsize=6.8,
                        color="black" if ref else c, fontweight="bold", zorder=10,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
            if len(x) >= args.min_points:
                fit = fit_curve(np, x, y, kind=args.fit, degree=args.degree)
                if fit:
                    fx, fy, r2 = fit
                    ax.plot(fx, fy, "-", color=c, lw=3.2 if ref else 2.0,
                            alpha=0.95, zorder=5 if ref else 2)
                    panel_r2.append((name, r2))
                    if pid == "kl_vs_prior":
                        fit_report.append((name, len(x), r2))
                    continue
            ax.plot(x, y, "-", color=c, lw=1.0, alpha=0.5)

        if base is not None and base.get(yk) is not None:
            bx = 0.0 if xk == args.kl_key else base.get(xk)
            if bx is not None:
                ax.scatter([bx], [base[yk]], marker="*", s=380, color="black", zorder=7)
            ax.axhline(base[yk], ls=":", c="0.5", lw=1.2, alpha=0.8)

        # R^2 differs per panel, so it cannot live in the figure-wide legend. Report the median
        # per-config fit quality on the panel it actually describes.
        if panel_r2:
            med = sorted(r for _, r in panel_r2)[len(panel_r2) // 2]
            ax.text(0.02, 0.97, f"median per-config $R^2$ = {med:.2f}  (n={len(panel_r2)})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="0.3")

        # the paper's claim is a SINGLE curve across methods — show it where that claim lives
        if pid == "kl_vs_prior" and len(allx) >= 4:
            X, Y = np.array(allx), np.array(ally)
            fit = fit_curve(np, X, Y, kind=args.fit, degree=args.degree)
            if fit:
                fx, fy, r2 = fit
                ax.plot(fx, fy, "--", color="0.2", lw=2.6, alpha=0.85,
                        label=f"pooled fit  $R^2$={r2:.2f}", zorder=6)
                fit_report.append(("POOLED", len(X), r2))
                # A near-zero pooled R^2 here is the headline of this panel, and a reader skimming
                # the figure will otherwise mistake a flat cloud for "no forgetting happened".
                if r2 < 0.25:
                    ax.text(0.5, 0.03, f"no KL-forgetting relationship (pooled $R^2$={r2:.3f}) — the "
                            f"retention probes\nare too small to resolve it, not evidence that "
                            f"forgetting is absent",
                            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                            color="0.25", bbox=dict(fc="#fff6e5", ec="#e0b070", lw=0.8, pad=4))

        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.25)
        if invert_x:
            ax.invert_xaxis()  # NLL: lower = better, so flip to read left-to-right as "more learned"
            # The cold variant-a runs never learn the task, so their NLL runs off to ~2.7 and squashes
            # every other curve into the right-hand sliver. Crop to where the runs that DID learn
            # live, and name the excluded ones on the panel — cropping silently would read as though
            # those configurations had simply not been tried.
            keep = [r[xk] for n in names for r in configs[n]
                    if r.get(xk) is not None and configs[n][0].get("variant") != "a"]
            if keep and base is not None and base.get(xk) is not None:
                keep.append(base[xk])
            if keep:
                cut = max(keep) * 1.06
                off = sorted({n for n in names for r in configs[n]
                              if r.get(xk) is not None and r[xk] > cut},
                             key=lambda n: configs[n][0].get("temperature") or 0)
                ax.set_xlim(cut, min(keep) * 0.985)
                if off:
                    ax.text(0.5, 0.035,
                            "cropped: " + ", ".join(disp(n) for n in off) +
                            f"\nextend past NLL {cut:.2f} (up to {max(r[xk] for n in off for r in configs[n]):.2f}) — "
                            "too cold to learn the task at all",
                            transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                            color="0.25", bbox=dict(fc="#fff6e5", ec="#e0b070", lw=0.8, pad=3))
    axes[2].invert_yaxis()

    from matplotlib.lines import Line2D
    handles = []
    for n in names:
        if len(configs[n]) == 1:
            handles.append(Line2D([], [], color=colors[n], marker=markers[n], ms=8, ls="none",
                                  mfc="none", mew=2.0, label=f"{n} (1 ckpt)"))
            continue
        is_ref = configs[n][0].get("variant") is None
        handles.append(Line2D([], [], color=colors[n], lw=3.2 if is_ref else 2.0,
                              marker=markers[n], ms=6 if is_ref else 5,
                              label=disp(n)))
    handles.append(Line2D([], [], color="black", marker="*", ms=13, ls="none", label="base $\\pi_0$"))
    # Shape is the second, colour-independent encoding, so state it rather than leaving it to be
    # inferred from twelve run labels.
    handles += [Line2D([], [], color="0.35", marker="s", ms=7, ls="none",
                       label="$\\bf{square}$ = variant a"),
                Line2D([], [], color="0.35", marker="o", ms=7, ls="none",
                       label="$\\bf{circle}$ = variant b")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fit_desc = "monotone (isotonic) fit" if args.fit == "isotonic" else f"degree-{args.degree} fit"
    fig.suptitle(f"KL vs forgetting across Impl-3 configurations — {sum(len(v) for v in configs.values())} "
                 f"checkpoints from {len(configs)} runs ({fit_desc} per configuration)\n"
                 f"KL measured {kl_cond};  prior task = {args.prior_key}", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = os.path.join(out_dir, f"fig3_kl_forgetting{args.suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")

    print(f"{sum(len(v) for v in configs.values())} checkpoints across {len(configs)} configs")
    print(f'{"config":<14}{"n":>4}{"R2 (prior ~ KL)":>18}')
    print("-" * 36)
    for name, n, r2 in fit_report:
        print(f"{disp(name):<14}{n:>4}{r2:>18.3f}")
    print(f"\nsaved {path}")

    if not args.no_robustness:
        plot_kl_robustness(np, plt, pe, args, configs, names, colors, markers, base,
                           prior_label, out_dir)


if __name__ == "__main__":
    main()
