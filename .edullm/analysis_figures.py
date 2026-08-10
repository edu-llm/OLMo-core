#!/usr/bin/env python3
"""The four figures the tranche's result is presented as, drawn from the analysis and nothing else.

SEPARATE FROM ``analysis.py`` FOR THE REASON ``noise_floor.py`` IS SEPARATE FROM
``wandb_panels.py``. A chart cannot fail because a variance did not converge, an estimator
cannot fail because matplotlib is not installed, and neither file grows a second purpose. This
module holds no arithmetic that decides anything: every number it draws is read out of what
:func:`analysis.analyse` already computed, so a figure and the report it sits beside cannot
disagree.

FOUR FIGURES, AND EACH ONE ANSWERS A QUESTION A READER WILL ASK OUT LOUD.

``loss-curves``      What did the runs do. Held-out bits-per-byte against step, one colour per
                     arm, with **every seed drawn** and a band over their range -- not a mean
                     with a standard error, which at five seeds is a summary of a picture small
                     enough to show whole. Beside it the training cross-entropy, which is where
                     a loss spike would be visible if one got past the optimizer.
``endpoint``         Is the effect bigger than the thing that limits it. The five endpoints of
                     each arm as points, the arm mean with its interval, and behind them a band
                     of plus and minus two pooled sigma around the baseline -- the measured
                     noise floor, drawn to the same scale as the effect, so the comparison the
                     gate makes is one the eye can make too.
``per-source``       Is the effect in one source or in all seven. Arm means per source, and
                     beneath them the difference from the comparator with its interval, so a
                     result that lives entirely in ``starcoder`` cannot be averaged into a
                     result about language modelling.
``stability``        H7. Declined optimizer steps per run and the largest gradient norm that
                     triggered one, both as every-run dot plots with the exact permutation
                     p-value printed on the panel, and the floor of that p-value printed under
                     it so that a null is read as the weak statement it is.

A SYNTHETIC FIGURE IS STAMPED ACROSS ITS FACE AND IT IS NOT SUBTLE. The report's synthetic
banner scrolls; a figure gets pasted into a slide, and by then there is no banner. So
:func:`draw` puts a rotated ``SYNTHETIC -- NOT A MEASUREMENT`` across every axis, writes the
files under a ``synthetic-`` prefix, and repeats the label in the provenance footer that every
figure carries anyway.
"""

import math
import os
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

#: One colour per arm, held here rather than left to the cycler so that ``baseline`` is the same
#: grey in all four figures and the treatment arms keep their identity across a slide deck.
PALETTE: Dict[str, str] = {
    "baseline": "#4d4d4d",
    "faithful": "#0b6fa4",
    "output-only": "#c1651a",
    "mhc": "#5b8c2a",
}

MARKERS: Dict[str, str] = {
    "baseline": "o",
    "faithful": "s",
    "output-only": "^",
    "mhc": "D",
}

#: One colour and marker per hypothesis, for the panels whose rows are contrasts rather than
#: arms.
#:
#: KEYED ON THE HYPOTHESIS AND NOT DERIVED FROM THE ARMS, because no rule over the arms works.
#: H1 and H2a share ``faithful`` as their treatment, so colouring by treatment merges them; H1
#: and H5 share ``baseline`` as their comparator, so colouring by "the arm that is not
#: ``faithful``" merges those two instead and paints H1 in the comparator's grey, which reads
#: as though the baseline were the thing being plotted. Three contrasts over four arms have no
#: single distinguishing arm and the honest fix is to stop looking for one.
HYPOTHESIS_PALETTE: Dict[str, str] = {
    "H1": "#0b6fa4",
    "H2a": "#c1651a",
    "H5": "#5b8c2a",
}

HYPOTHESIS_MARKERS: Dict[str, str] = {"H1": "s", "H2a": "^", "H5": "D"}

#: ``BPB = CE_nats / (bytes_per_token * ln 2)``, the same constant ``noise_floor`` carries.
NATS_PER_BPB = 4.57 * math.log(2)


def _style():
    """The house style, applied once per figure rather than set globally on import."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.6,
        }
    )
    return plt


def _provenance(result: Mapping[str, object]) -> str:
    """One line naming what produced every number in the figure it is written under."""
    arms = result.get("arms") or []
    named = ", ".join(f"{a['arm']}={a['submission']}" for a in arms)  # type: ignore[index]
    stamp = f"{result.get('label', 'measured')}, generated {result.get('generated')}"
    provisional = " -- PROVISIONAL" if result.get("provisional") else ""
    return f"{stamp}{provisional} | {named}"


def _finish(fig, result: Mapping[str, object], path: str) -> str:
    """Stamp, caption and write one figure."""
    label = str(result.get("label", "measured"))
    fig.text(
        0.5,
        0.005,
        _provenance(result),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#666666",
    )
    if label != "measured":
        for axis in fig.get_axes():
            axis.text(
                0.5,
                0.5,
                "SYNTHETIC\nNOT A MEASUREMENT",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=17,
                color="#d62728",
                alpha=0.24,
                rotation=27,
                zorder=50,
                fontweight="bold",
            )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    return path


def _ordered(arms: Sequence, result: Mapping[str, object]) -> List:
    order = [a["arm"] for a in result["arms"]]  # type: ignore[index]
    by_name = {arm.arm: arm for arm in arms}
    return [by_name[name] for name in order if name in by_name]


def loss_curves(arms: Sequence, result: Mapping[str, object], path: str) -> str:
    """
    Held-out bits-per-byte against step, and the training curve beside it.

    EVERY SEED IS DRAWN. A mean with a ribbon at five replicates is a summary of something small
    enough to show, and the thing a reader most wants to know -- whether one seed is carrying
    the arm -- is exactly what the ribbon removes. The band is the range over seeds and not a
    standard error, so it is a statement about the runs rather than about a sampling
    distribution.

    :param arms: The arms as read.
    :param result: What ``analysis.analyse`` returned.
    :param path: Where to write the PNG. A PDF goes beside it.

    :returns: The path written.
    """
    plt = _style()
    ordered = _ordered(arms, result)
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14.4, 4.3))

    for arm in ordered:
        colour = PALETTE.get(arm.arm, "#888888")
        curves = np.asarray(arm.bpb, dtype=float).mean(axis=2)
        steps = np.asarray(arm.steps, dtype=float)
        left.fill_between(
            steps, curves.min(axis=0), curves.max(axis=0), color=colour, alpha=0.18, linewidth=0
        )
        for row in curves:
            left.plot(steps, row, color=colour, alpha=0.4, linewidth=0.7)
        left.plot(
            steps,
            curves.mean(axis=0),
            color=colour,
            marker=MARKERS.get(arm.arm, "o"),
            markersize=3.2,
            label=f"{arm.arm} (n={curves.shape[0]})",
        )
    # Clipped to the range after the first evaluation. Step 0 is an untrained model at 3.7 BPB
    # against an endpoint near 0.68, so on an axis that includes it the whole run after step 500
    # is four pixels of flat line and the panel says only that training happened.
    after_first = [
        float(v)
        for arm in ordered
        for step, v in zip(arm.steps, np.asarray(arm.bpb, dtype=float).mean(axis=2).T.tolist())
        if step > 0
        for v in v
    ]
    if after_first:
        low, high = min(after_first), max(after_first)
        margin = 0.08 * (high - low) or 0.01
        left.set_ylim(low - margin, high + margin)
    left.set_xlabel("training step (of 6,000; 4.72B dolma2 tokens)")
    left.set_ylabel("held-out bits-per-byte, unweighted mean of 7 sources")
    left.set_title(
        "The runs, from the first evaluation after step 0\n"
        "band is the range over seeds, not a standard error"
    )
    left.legend(loc="upper right", fontsize=7.5)

    # THE MIDDLE PANEL IS THE ONE WORTH LOOKING AT, AND THE LEFT ONE IS WHY. The curve falls
    # by about 0.6 BPB over the run and the effect the tranche is hunting is 0.003, so on an
    # absolute axis the four arms are one line and the only thing a reader learns is that they
    # all trained. Subtracting the baseline *at the same seed* removes the training trend and
    # the shared data order at once, which leaves exactly the quantity the primary analysis
    # tests, plotted against step so that "the effect appears late" and "there is no effect"
    # are distinguishable rather than both reading as a flat line at the end.
    # ARM BY ARM ON THE STEPS THAT ARM SHARES WITH THE BASELINE, rather than on one grid for
    # all of them. Mid-tranche the arms are different ages -- the comparator has thirteen
    # evaluations and a treatment three hours behind it has seven -- so a difference taken
    # position by position subtracts step 6,000 from step 3,000 and draws the training curve as
    # though it were an effect. That is the same mistake the contrast made before it was
    # aligned, and here it would be drawn rather than printed.
    base = next((a for a in ordered if a.arm == "baseline"), None)
    if base is not None and len(ordered) > 1:
        reference = np.asarray(base.bpb, dtype=float).mean(axis=2)
        for arm in ordered:
            if arm.arm == "baseline":
                continue
            colour = PALETTE.get(arm.arm, "#888888")
            shared = [step for step in arm.steps if step in base.steps]
            if not shared:
                continue
            mine = np.asarray(arm.bpb, dtype=float).mean(axis=2)
            theirs = reference[:, [base.steps.index(step) for step in shared]]
            mine = mine[:, [arm.steps.index(step) for step in shared]]
            differences = (mine - theirs) * NATS_PER_BPB
            steps = np.asarray(shared, dtype=float)
            middle.fill_between(
                steps,
                differences.min(axis=0),
                differences.max(axis=0),
                color=colour,
                alpha=0.18,
                linewidth=0,
            )
            middle.plot(
                steps,
                differences.mean(axis=0),
                color=colour,
                marker=MARKERS.get(arm.arm, "o"),
                markersize=3.2,
                label=f"{arm.arm} - baseline",
            )
        floor = float(result["sigma"]["sigma_bpb_unbiased"]) * NATS_PER_BPB  # type: ignore[index]
        gate = 2.0 * floor * math.sqrt(2.0 / max(len(ordered[0].seeds), 1))
        middle.axhspan(
            -gate, gate, color="#9ecae1", alpha=0.28, zorder=0, label="the gate, +/- 2 SE"
        )
        middle.axhline(0.0, color="#333333", linewidth=1.0)
        middle.set_xlabel("training step")
        middle.set_ylabel(
            "paired difference from baseline\n(nats of held-out CE, negative is better)"
        )
        middle.set_title(
            "The effect, paired on seed\nband is the range over the five paired differences"
        )
        middle.legend(loc="upper right", fontsize=7.5)
    else:
        middle.set_axis_off()
        middle.text(
            0.5,
            0.5,
            "no baseline in this read, so there is no paired difference to draw",
            ha="center",
            va="center",
            transform=middle.transAxes,
            color="#888888",
            wrap=True,
        )

    drew_train = False
    for arm in ordered:
        colour = PALETTE.get(arm.arm, "#888888")
        for steps, loss in zip(arm.train_curve_steps, arm.train_curve_loss):
            if len(steps) < 2:
                continue
            right.plot(steps, loss, color=colour, alpha=0.55, linewidth=0.7)
            drew_train = True
    if drew_train:
        right.set_yscale("log")
        right.set_xlabel("training step")
        right.set_ylabel("training cross-entropy (nats)")
        right.set_title(
            "Training loss, every seed\nwhere a spike would show if one got past the optimizer"
        )
        right.legend(
            handles=[
                plt.Line2D([], [], color=PALETTE.get(a.arm, "#888888"), label=a.arm)
                for a in ordered
            ],
            loc="upper right",
            fontsize=7.5,
        )
    else:
        right.set_axis_off()
        right.text(
            0.5,
            0.5,
            "no training curve in this read",
            ha="center",
            va="center",
            transform=right.transAxes,
            color="#888888",
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _finish(fig, result, path)


def _at_step(arm, result: Mapping[str, object]) -> np.ndarray:
    """
    One arm's ``(n_seeds, n_sources)`` bits-per-byte at the step the analysis compared at.

    NOT ``[:, -1, :]``, WHICH IS WHERE THIS FIGURE DREW A RESULT THAT WAS NOT THERE. Each arm's
    own last evaluation is a different step whenever the arms are different ages, so a
    mid-tranche endpoint chart put the finished comparator at step 6,000 and a treatment three
    hours behind it at step 3,000, and drew the half of the training curve between them as the
    gap between two arms. The contrast beside it had already been aligned; the figure had not,
    so the two disagreed by a factor of thirty and the figure was the one people would have
    looked at.

    :param arm: An ``analysis.ArmSeries``.
    :param result: What ``analysis.analyse`` returned, for ``compared_at_step``.

    :returns: The matrix.
    """
    step = result.get("compared_at_step")
    index = arm.steps.index(step) if step in arm.steps else -1
    return np.asarray(arm.bpb, dtype=float)[:, index, :]


def endpoint(arms: Sequence, result: Mapping[str, object], path: str) -> str:
    """
    The endpoint of every run, per arm, against the measured noise floor.

    THE POINT OF THE FIGURE IS THE BAND AND NOT THE POINTS. An effect is only a result if it is
    bigger than the quantity that limits it, and every version of this chart that plots arm
    means alone invites the reader to compare a gap to the width of a marker. So the pooled
    sigma is drawn at the same scale as the effect: a shaded band of plus and minus two sigma
    around the baseline mean, which is the pre-registered gate, with the minimum detectable
    effect marked beside it.

    :param arms: The arms as read.
    :param result: What ``analysis.analyse`` returned.
    :param path: Where to write the PNG.

    :returns: The path written.
    """
    plt = _style()
    ordered = _ordered(arms, result)
    sigma = result["sigma"]  # type: ignore[index]
    fig, axis = plt.subplots(figsize=(7.6, 4.6))

    baseline = next((a for a in result["arms"] if a["arm"] == "baseline"), None)  # type: ignore[index]
    centre = (
        float(baseline["mean_bpb"])
        if baseline
        else float(np.mean([a["mean_bpb"] for a in result["arms"]]))  # type: ignore[index]
    )
    gate = 2.0 * float(sigma["sigma_bpb_unbiased"]) * math.sqrt(2.0 / max(len(ordered[0].seeds), 1))
    floor = float(sigma["sigma_bpb_unbiased"])

    axis.axhspan(centre - gate, centre + gate, color="#9ecae1", alpha=0.28, zorder=0)
    axis.axhspan(centre - floor, centre + floor, color="#6b6b6b", alpha=0.22, zorder=0)
    axis.axhline(centre, color="#4d4d4d", linewidth=0.9, linestyle="--", zorder=1)
    axis.plot(
        [],
        [],
        color="#6b6b6b",
        alpha=0.4,
        linewidth=9,
        label=f"+/- 1 pooled sigma-hat: {floor * NATS_PER_BPB:.4f} nats, df {sigma['df']}",
    )
    axis.plot(
        [],
        [],
        color="#9ecae1",
        alpha=0.6,
        linewidth=9,
        label=f"the gate, +/- 2 SE of a 5 v 5 contrast: {gate * NATS_PER_BPB:.4f} nats",
    )

    primary = {
        entry["name"]: next((r for r in entry.get("rows", []) if r["primary"]), None)
        for entry in result.get("contrasts", [])  # type: ignore[union-attr]
    }
    # Keyed by treatment and NOT overwritten, because H1 and H2a share `faithful` as the
    # treatment and the second would silently replace the first -- which put H2a's number under
    # the `faithful` tick with no comparator named beside it, and that is a mislabelled figure
    # rather than a missing one.
    by_treatment: Dict[str, tuple] = {}
    by_comparator: Dict[str, tuple] = {}
    for entry in result.get("contrasts", []):  # type: ignore[union-attr]
        row = primary.get(entry.get("name", ""))
        if not row:
            continue
        if entry.get("treatment") and entry["treatment"] not in by_treatment:
            by_treatment[entry["treatment"]] = (entry["name"], entry["comparator"], row, 1.0)
        if entry.get("comparator") and entry["comparator"] not in by_comparator:
            by_comparator[entry["comparator"]] = (entry["name"], entry["treatment"], row, -1.0)

    positions = np.arange(len(ordered), dtype=float)
    rng = np.random.default_rng(0)
    for position, arm in zip(positions, ordered):
        colour = PALETTE.get(arm.arm, "#888888")
        values = _at_step(arm, result).mean(axis=1)
        jitter = position + rng.uniform(-0.11, 0.11, values.size)
        axis.scatter(
            jitter,
            values,
            color=colour,
            marker=MARKERS.get(arm.arm, "o"),
            s=34,
            zorder=4,
            edgecolor="white",
            linewidth=0.6,
        )
        for seed, x, y in zip(arm.seeds, jitter, values):
            axis.annotate(
                str(seed),
                (x, y),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=6,
                color=colour,
            )
        mean = float(values.mean())
        half = float(values.std(ddof=1)) / math.sqrt(values.size) * 2.776  # t_{.975,4}
        axis.errorbar(
            position,
            mean,
            yerr=half,
            color=colour,
            capsize=5,
            elinewidth=1.6,
            marker="_",
            markersize=26,
            markeredgewidth=2.2,
            zorder=5,
        )

    labels = []
    for arm in ordered:
        # An arm that is only ever a comparator -- `output-only` is H2a's -- would otherwise get
        # a bare tick and sit in the middle of the panel with no number on it. Its contrast is
        # written from its own side, with the sign turned round so the label means what it says.
        # The arm the whole panel is centred on is left alone: the dashed line through it and the
        # right-hand axis already say it is the zero, and H1 read backwards under its tick is the
        # same number twice.
        entry = by_treatment.get(arm.arm)
        if entry is None and arm.arm != (baseline or {}).get("arm"):
            entry = by_comparator.get(arm.arm)
        if entry is None:
            labels.append(f"{arm.arm}\nn={len(arm.seeds)}")
            continue
        name, against, row, sign = entry
        verdict = "clears the gate" if row["clears_gate"] else "inside the gate"
        labels.append(
            f"{arm.arm}\nn={len(arm.seeds)}\n{name} vs {against}\n"
            f"{sign * row['delta_nats']:+.4f} nats, {verdict}"
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7.5)
    axis.set_ylabel(
        f"held-out bits-per-byte at step {result.get('compared_at_step'):,}\n"
        "(unweighted mean of 7 sources)"
    )
    secondary = axis.secondary_yaxis(
        "right",
        functions=(
            lambda v: (v - centre) * NATS_PER_BPB,
            lambda v: v / NATS_PER_BPB + centre,
        ),
    )
    secondary.set_ylabel("nats of held-out cross-entropy,\nrelative to the baseline mean")
    axis.set_title(
        "Endpoint per run against the measured noise floor\n"
        "points are runs; bars are the arm mean with a 95% t interval on 4 df"
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return _finish(fig, result, path)


def per_source(arms: Sequence, result: Mapping[str, object], path: str) -> str:
    """
    Bits-per-byte per held-out source, and the per-source difference from the comparator.

    AN AVERAGE OVER ARXIV, CODE, WEB TEXT AND WIKIPEDIA IS EXACTLY THE KIND THAT HIDES THE
    EFFECT IT IS MEANT TO MEASURE, which is the pre-registration's own reason for taking the
    publisher's stratified validation split rather than carving one. The top panel is the level
    per source, which is mostly a statement about the corpus; the bottom panel is the contrast
    per source with its interval, which is where an effect concentrated in one source shows up
    as one bar away from zero rather than as a seventh of a pooled mean.

    :param arms: The arms as read.
    :param result: What ``analysis.analyse`` returned.
    :param path: Where to write the PNG.

    :returns: The path written.
    """
    plt = _style()
    ordered = _ordered(arms, result)
    sources = list(ordered[0].sources)
    entries = list(result.get("per_source") or [])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10.0, 6.6), gridspec_kw={"height_ratios": [1.0, 1.15]}
    )

    width = 0.72 / max(len(ordered), 1)
    base = np.arange(len(sources), dtype=float)
    for index, arm in enumerate(ordered):
        colour = PALETTE.get(arm.arm, "#888888")
        matrix = _at_step(arm, result)
        offset = base + (index - (len(ordered) - 1) / 2.0) * width
        # Points and not bars, so the axis can be truncated to the data honestly. A bar chart
        # has to start at zero or it lies, and starting at zero on a range of 0.33 to 1.05 BPB
        # puts every arm on top of every other one.
        top.vlines(offset, matrix.min(axis=0), matrix.max(axis=0), color=colour, linewidth=1.1)
        top.scatter(
            offset,
            matrix.mean(axis=0),
            color=colour,
            marker=MARKERS.get(arm.arm, "o"),
            s=26,
            zorder=4,
            edgecolor="white",
            linewidth=0.5,
            label=arm.arm,
        )
    top.set_xticks(base)
    top.set_xticklabels(sources, rotation=12)
    top.set_ylabel("held-out bits-per-byte")
    top.set_title(
        f"Per source at step {result.get('compared_at_step'):,} -- marker is the arm mean, "
        "whisker is the range over 5 seeds"
    )
    # Headroom before the legend goes in, because dclm sits near 1.03 BPB and an upper-left
    # legend on the default limits lands on top of it -- losing the highest-loss source from the
    # panel whose whole subject is which source moved.
    low, high = top.get_ylim()
    top.set_ylim(low, high + 0.22 * (high - low))
    top.legend(loc="upper left", ncol=len(ordered))

    if entries:
        span = 0.72 / max(len(entries), 1)
        for index, entry in enumerate(entries):
            rows = {r["source"]: r for r in entry["rows"]}
            deltas = np.asarray([rows[s]["delta_nats"] for s in sources], dtype=float)
            errors = np.asarray(
                [1.96 * rows[s]["se_bpb"] * NATS_PER_BPB for s in sources], dtype=float
            )
            contrast = next(
                (c for c in result["contrasts"] if c["name"] == entry["name"]),  # type: ignore[index]
                {},
            )
            name = str(entry["name"])
            colour = HYPOTHESIS_PALETTE.get(name, "#888888")
            offset = base + (index - (len(entries) - 1) / 2.0) * span
            bottom.errorbar(
                offset,
                deltas,
                yerr=errors,
                fmt=HYPOTHESIS_MARKERS.get(name, "o"),
                color=colour,
                capsize=3,
                markersize=5,
                elinewidth=1.3,
                label=f"{entry['name']}: {contrast.get('treatment')} - {contrast.get('comparator')}",
            )
        bottom.axhline(0.0, color="#333333", linewidth=1.0)
        bottom.set_xticks(base)
        bottom.set_xticklabels(sources, rotation=12)
        bottom.set_ylabel("difference from comparator\n(nats of held-out CE, negative is better)")
        bottom.set_title(
            "Per-source contrast, paired on seed, with a 95% interval on the blocked error " "term",
            pad=26,
        )
        # Outside the axes. Inside, the legend's own sample marker sits at the top left of a
        # panel whose leftmost source is also near zero, and it reads as an eighth data point.
        bottom.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(len(entries), 1),
            fontsize=7.5,
            frameon=False,
        )
    else:
        bottom.set_axis_off()
        bottom.text(
            0.5,
            0.5,
            "no contrast in this read: only one arm landed",
            ha="center",
            va="center",
            transform=bottom.transAxes,
            color="#888888",
        )

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return _finish(fig, result, path)


def stability(arms: Sequence, result: Mapping[str, object], path: str) -> str:
    """
    H7: declined optimizer steps and the gradient norms that triggered them.

    BOTH PANELS, BECAUSE THE COUNT ALONE DOES NOT SEPARATE THE TWO THINGS H7 IS ABOUT. A run
    that declined a dozen unremarkable updates and a run that declined the onset of a spike have
    similar counts and completely different largest triggers -- measured on the comparator, the
    largest trigger on a clean run was 0.712 against 9.30 and 20.45 on the two that spiked under
    plain AdamW. The exact permutation p is printed on each panel, and its floor is printed
    under it, because a p of 0.09 from a 5 v 5 permutation test means "these did not separate
    completely" and not "these are similar".

    :param arms: The arms as read.
    :param result: What ``analysis.analyse`` returned.
    :param path: Where to write the PNG.

    :returns: The path written.
    """
    plt = _style()
    ordered = _ordered(arms, result)
    tests = {t["arm"]: t for t in (result.get("h7") or {}).get("tests", []) if "arm" in t}  # type: ignore[union-attr]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.4))

    # Counted through the step the analysis compared at, not through whatever step each arm
    # happened to reach. A cumulative count over a longer run is a bigger count, so plotting
    # each arm's own running total draws a treatment that is three hours behind as the calmer
    # of the two -- and the panel would then disagree with the p-value printed on it.
    through = result.get("h7", {}).get("through_step")  # type: ignore[union-attr]
    positions = np.arange(len(ordered), dtype=float)
    rng = np.random.default_rng(1)
    for axis, extract, ylabel, key in (
        (
            left,
            lambda a: a.stability_at(through)[0],
            f"declined optimizer steps in the first {through:,}",
            "primary",
        ),
        (
            right,
            lambda a: a.stability_at(through)[1],
            f"largest gradient norm at a declined step,\nin the first {through:,} steps",
            "secondary",
        ),
    ):
        for position, arm in zip(positions, ordered):
            values = np.asarray(extract(arm), dtype=float)
            if not values.size:
                continue
            colour = PALETTE.get(arm.arm, "#888888")
            jitter = position + rng.uniform(-0.12, 0.12, values.size)
            # A seed that declined nothing has no largest trigger, and it is carried as 0.0 --
            # which a log axis cannot draw, so it would vanish and take a replicate out of a
            # panel whose whole subject is how many replicates fell where. Drawn hollow at the
            # floor instead, and counted in the caption.
            zero = values <= 0.0
            axis.scatter(
                jitter[~zero],
                values[~zero],
                color=colour,
                marker=MARKERS.get(arm.arm, "o"),
                s=40,
                zorder=4,
                edgecolor="white",
                linewidth=0.6,
            )
            if zero.any() and key == "secondary":
                lowest = float(np.min(values[values > 0.0])) if (values > 0.0).any() else 1.0
                axis.scatter(
                    jitter[zero],
                    np.full(int(zero.sum()), lowest * 0.72),
                    facecolor="none",
                    edgecolor=colour,
                    marker=MARKERS.get(arm.arm, "o"),
                    s=40,
                    zorder=4,
                    linewidth=1.0,
                    label=f"{arm.arm}: declined nothing, so no trigger to report",
                )
                axis.legend(loc="lower left", fontsize=6.5, frameon=False)
            axis.hlines(
                float(values.mean()),
                position - 0.24,
                position + 0.24,
                color=colour,
                linewidth=2.2,
                zorder=5,
            )
            test = tests.get(arm.arm)
            item = (test or {}).get(key)
            if item:
                tag = "H7" if test.get("pre_registered") else "post-hoc"
                axis.annotate(
                    f"p = {item['p_value']:.4f}\n[{tag}]",
                    (position, float(values.max())),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=7,
                    color=colour,
                )
        axis.set_xticks(positions)
        axis.set_xticklabels([a.arm for a in ordered])
        axis.set_ylabel(ylabel)
        axis.margins(y=0.22)

    right.set_yscale("log")
    left.set_title(
        f"H7 primary: declined updates in the first {through:,} steps\n"
        "exact two-sided permutation test against the baseline"
    )
    right.set_title(
        "H7 secondary: largest triggering gradient norm\nthe statistic that separates a spike from noise"
    )
    floor = 2.0 / 252.0
    fig.text(
        0.5,
        0.045,
        f"5 v 5 permutation: the smallest attainable two-sided p is 2/C(10,5) = {floor:.4f}. "
        "Complete separation is detectable; partial separation mostly is not.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return _finish(fig, result, path)


def equal_compute(arms: Sequence, result: Mapping[str, object], path: str) -> str:
    """
    The same curves against wall clock instead of against steps, where the ordering reverses.

    THE X AXIS IS THE WHOLE POINT. Every other figure here puts the step count on the bottom,
    which is the pre-registered comparison and the one the two papers argue about. A reader who
    sees only those figures will conclude the mechanism is worth having, because at equal steps
    it plainly is. Drawn against hours on the same machine, the arm that costs 1.7x per step is
    behind for the entire run and the conclusion inverts. Both are true; showing only one is the
    misleading choice, so this is a figure rather than a paragraph.

    :param arms: The arms as read.
    :param result: What ``analysis.analyse`` returned.
    :param path: Where to write the PNG.

    :returns: The path written.
    """
    plt = _style()
    ordered = _ordered(arms, result)
    rows = {str(r["arm"]): r for r in (result.get("equal_compute") or [])}
    fig, axis = plt.subplots(figsize=(7.8, 4.8))

    budget = None
    for arm in ordered:
        hours, values, spread = [], [], []
        for step in arm.steps:
            seconds = arm.seconds_at(step)
            if seconds is None or step == 0:
                continue
            column = np.asarray(arm.bpb, dtype=float)[:, arm.steps.index(step), :].mean(axis=1)
            hours.append(seconds / 3600.0)
            values.append(float(column.mean()) * NATS_PER_BPB)
            spread.append((float(column.min()) * NATS_PER_BPB, float(column.max()) * NATS_PER_BPB))
        if not hours:
            continue
        colour = PALETTE.get(arm.arm, "#888888")
        axis.plot(
            hours,
            values,
            color=colour,
            marker=MARKERS.get(arm.arm, "o"),
            markersize=3.4,
            linewidth=1.5,
            label=f"{arm.arm} (n={len(arm.seeds)})",
        )
        axis.fill_between(
            hours,
            [s[0] for s in spread],
            [s[1] for s in spread],
            color=colour,
            alpha=0.16,
            linewidth=0,
        )
        if arm.arm == "baseline":
            budget = hours[-1]

    if budget is not None:
        axis.axvline(budget, color="#4d4d4d", linestyle="--", linewidth=1.0)
        told = [
            f"the baseline finishes its 6,000 steps at {budget:.1f} h,",
            "and in that same budget",
        ]
        told += [
            f"    {name} is {float(row['gap_nats']):+.4f} nats behind" for name, row in rows.items()
        ]
        axis.annotate(
            "\n".join(told),
            (budget, 0.42),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            fontsize=7.8,
            color="#333333",
        )
    axis.set_xlabel("wall clock on the same 8xA100 node (hours)")
    axis.set_ylabel("held-out cross-entropy (nats)\nunweighted mean of 7 sources")
    axis.set_title(
        "At equal compute the ordering reverses\n"
        "band is the range over seeds; the pre-registered endpoint is at equal STEPS, not here"
    )
    axis.legend(loc="upper right")
    return _finish(fig, result, path)


def draw(
    arms: Sequence,
    result: Mapping[str, object],
    out: str,
    only: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Draw every figure into one directory, prefixed so a synthetic run cannot be mistaken.

    :param arms: The arms as read.
    :param result: What ``analysis.analyse`` returned.
    :param out: Directory to write into. Created if absent.
    :param only: Restrict to these figure names.

    :returns: The paths written.
    """
    prefix = "" if result.get("label") == "measured" else "synthetic-"
    wanted = {
        "loss-curves": loss_curves,
        "endpoint": endpoint,
        "per-source": per_source,
        "stability": stability,
        "equal-compute": equal_compute,
    }
    written = []
    for name, function in wanted.items():
        if only and name not in only:
            continue
        if name == "equal-compute" and not result.get("equal_compute"):
            continue
        written.append(function(arms, result, os.path.join(out, f"{prefix}{name}.png")))
    return written
