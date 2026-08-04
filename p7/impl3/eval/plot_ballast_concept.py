#!/usr/bin/env python
"""Conceptual figures for Ballast SFT, in the spirit of Figure 1 of RL's Razor.

Four scenes (standalone + one merged 2x2 figure):

    ballast_concept_1_sft.png       standard SFT: every token carries equal mass
    ballast_concept_2_klpenalty.png adding a KL term to the loss: the SOLUTION is dragged
                                    back toward the base model, into the void, while the
                                    data itself is still weighted uniformly
    ballast_concept_3_ballast.png   Ballast: mass is redistributed ACROSS the data, so the
                                    centre of mass moves to the low-KL edge of the region
                                    and never leaves it
    ballast_concept_4_rl.png        RL's Razor: on-policy training only ever sees samples
                                    already near the base model, so the data itself lives
                                    in the low-KL edge (uniform sizes — no reweighting)

The distinction Ballast exists to draw: a KL penalty and Ballast both lower KL, but they
are not the same move. A penalty adds a second force that pulls the solution off the data
entirely, into the void between the base model and the good region. Ballast changes which
parts of the region the model is pulled toward, and a weighted mean of the data cannot
leave the data's own hull. RL's diagram is the limiting case where the data never left
the low-KL edge in the first place.

Mass is conserved under Ballast. Weights are normalised to mean 1, so heavy tokens are
drawn LARGER than the uniform-SFT token and light ones smaller — marker sizes share one
absolute scale across the SFT / KL-penalty / Ballast panels.

Positions are synthetic (fixed seed); these are diagrams of the mechanism, not measurements.

    python eval/plot_ballast_concept.py --out_dir out/figures
"""
import argparse
import os

BASE_XY = (0.0, 0.0)
S0 = 150.0                    # marker area of a token with weight 1

# Good region: a RADIAL band — spans KL (radius) much more than angle — with an obvious
# void between the base model and the inner edge. That void is the argument of panel (b).
TH_LO, TH_HI = -26.0, 26.0    # narrow tangential wrap (degrees)
R_MID, H_MAX = 3.05, 1.55     # centreline radius and half-thickness → KL ≈ 1.5 … 4.6

TOKEN_FC, TOKEN_EC = "#ffc94d", "#4a3200"   # amber tokens
COM_BALLAST = "#3b82f6"                     # Ballast's new centre of mass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="out/figures")
    p.add_argument("--temperature", type=float, default=0.40,
                   help="T for the Ballast figure. Lower = more mass moved. Chosen for "
                        "legibility, not calibrated to any measured run.")
    p.add_argument("--n_points", type=int, default=115)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--name", default="Ballast")
    p.add_argument("--region_label", default="behaviours that solve\nthe new task")
    p.add_argument("--merged_only", action="store_true",
                   help="Write only the combined figure, not the standalone ones.")
    return p.parse_args()


def band(np, u):
    """Radial band at normalised arc position u in [-1, 1].

    Returns (angle, centreline radius, half-thickness). Thickness vanishes at the tips so
    the ends round into a bean instead of square annular-wedge corners. Mild low-frequency
    wiggle on the centreline keeps the boundary from looking machined.
    """
    th = np.deg2rad(TH_LO + (TH_HI - TH_LO) * (u + 1) / 2)
    rm = R_MID + 0.18 * np.sin(1.5 * np.pi * u + 0.4) + 0.08 * np.cos(2.7 * np.pi * u - 0.8)
    h = H_MAX * (1 - u ** 2) ** 0.35 * (1 + 0.10 * np.sin(2.1 * np.pi * u + 0.7))
    return th, rm, h


def to_xy(np, r, th):
    return BASE_XY[0] + r * np.cos(th), BASE_XY[1] + r * np.sin(th)


def region_path(np):
    """Closed boundary of the good region, as (xs, ys)."""
    u = np.linspace(-1, 1, 400)
    th, rm, h = band(np, u)
    xo, yo = to_xy(np, rm + h, th)
    th2, rm2, h2 = band(np, u[::-1])
    xi, yi = to_xy(np, rm2 - h2, th2)
    return np.concatenate([xo, xi, xo[:1]]), np.concatenate([yo, yi, yo[:1]])


def sample_tokens(np, rng, n, *, r_frac=(-0.86, 0.86)):
    """Fill the band (or a radial sub-band via ``r_frac``). Rejection keeps areal density even."""
    lo, hi = r_frac
    out = []
    while len(out) < n:
        u = rng.uniform(-1, 1, 4 * n)
        th, rm, h = band(np, u)
        u = u[rng.uniform(0, H_MAX, len(u)) < h]
        th, rm, h = band(np, u)
        r = rm + h * rng.uniform(lo, hi, len(u))
        x, y = to_xy(np, r, th)
        out.extend(zip(x, y))
    return np.array(out[:n])


def limits(np):
    """Axis limits wide enough for the region, the base model and its label."""
    xs, ys = region_path(np)
    xs = np.append(xs, BASE_XY[0]); ys = np.append(ys, BASE_XY[1])
    return ((xs.min() - 1.25, xs.max() + 0.55), (ys.min() - 0.70, ys.max() + 0.70))


def label(ax, text, xy, *, dx=0, dy=0, ha="center", size=15, color="white", z=12):
    """Text that survives being drawn over pale tokens as well as the dark field."""
    import matplotlib.patheffects as pe
    ax.annotate(text, xy, textcoords="offset points", xytext=(dx, dy), ha=ha,
                fontsize=size, color=color, fontweight="bold", zorder=z,
                path_effects=[pe.withStroke(linewidth=3.8, foreground="black")])


def kl_cmap(np, plt):
    """viridis with the pale end trimmed off, so amber tokens read at every KL."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.70, 256)))


def draw_field(np, plt, ax):
    XLIM, YLIM = limits(np)
    X, Y = np.meshgrid(np.linspace(*XLIM, 460), np.linspace(*YLIM, 460))
    Z = np.hypot(X - BASE_XY[0], Y - BASE_XY[1])
    im = ax.contourf(X, Y, Z, levels=32, cmap=kl_cmap(np, plt))
    cs = ax.contour(X, Y, Z, levels=[1, 2, 3, 4, 5], colors="white",
                    linewidths=1.0, alpha=0.55)
    ax.clabel(cs, fmt=lambda v: f"KL = {v:g}", fontsize=12, colors="white")
    ax.scatter(*BASE_XY, marker="*", s=640, color="white", edgecolor="0.1",
               linewidths=1.1, zorder=10)
    label(ax, r"base model $\pi_0$", BASE_XY, dy=-30)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    return im


def draw_region(np, ax, text=None):
    """Outline the good region. ``text=None`` skips the label (merged figure: name it once)."""
    xs, ys = region_path(np)
    ax.fill(xs, ys, facecolor="white", alpha=0.11, zorder=3)
    ax.plot(xs, ys, ls="--", color="white", lw=2.4, alpha=0.95, zorder=4)
    if not text:
        return
    import matplotlib.patheffects as pe
    th, rm, h = band(np, np.array([0.0]))   # outer mid-band: nearest to the top-left label
    ax.annotate(text, xy=to_xy(np, float(rm[0] + h[0]), float(th[0])),
                xytext=(0.035, 0.965), textcoords="axes fraction", ha="left", va="top",
                fontsize=15, color="white", fontweight="bold", zorder=12,
                path_effects=[pe.withStroke(linewidth=3.8, foreground="black")],
                arrowprops=dict(arrowstyle="->", color="white", lw=1.8))


def mark_com(ax, com, *, filled=True, color="white"):
    ax.scatter(*com, marker="v", s=340,
               color=color if filled else "none",
               edgecolor="0.1" if filled else "white",
               linewidths=1.3 if filled else 2.8, zorder=11)


def inside(np, com):
    from matplotlib.path import Path
    xs, ys = region_path(np)
    return bool(Path(np.column_stack([xs, ys])).contains_point(com))


def main():
    args = parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # SFT / KL-penalty / Ballast share one point cloud that spans the full KL band.
    pts = sample_tokens(np, rng, args.n_points)
    s = np.hypot(pts[:, 0] - BASE_XY[0], pts[:, 1] - BASE_XY[1])
    m_uniform = np.ones_like(s)
    # Mean-1 normalisation IS the mass-conservation constraint: sum(m) == n, so whatever
    # drops below 1 funds what rises above — heavy markers can exceed S0.
    w = np.exp(-s / args.temperature)
    m_ballast = w / w.mean()

    # RL's Razor: on-policy samples live on the low-KL (inner) edge of the same region.
    # Uniform sizes — the point is where the data sits, not that it is reweighted.
    pts_rl = sample_tokens(np, rng, args.n_points, r_frac=(-0.92, -0.35))
    s_rl = np.hypot(pts_rl[:, 0] - BASE_XY[0], pts_rl[:, 1] - BASE_XY[1])

    def com_of(points, mass):
        kl = np.hypot(points[:, 0] - BASE_XY[0], points[:, 1] - BASE_XY[1])
        return ((mass[:, None] * points).sum(0) / mass.sum(),
                float((mass * kl).sum() / mass.sum()))

    com_u, kl_u = com_of(pts, m_uniform)
    com_b, kl_b = com_of(pts, m_ballast)
    com_rl, kl_rl = com_of(pts_rl, np.ones(len(pts_rl)))

    d = (np.array(BASE_XY) - com_u) / np.hypot(*(np.array(BASE_XY) - com_u))
    tip = np.array(BASE_XY) - d * 0.95           # in the void, well short of the region

    def tokens(ax, points, mass):
        ax.scatter(points[:, 0], points[:, 1], s=S0 * mass, c=TOKEN_FC, alpha=0.97,
                   edgecolor=TOKEN_EC, linewidths=0.8, zorder=6)

    def panel(ax, kind, *, region_label=None):
        """Draw one scene. Used both standalone and side by side."""
        im = draw_field(np, plt, ax)
        draw_region(np, ax, region_label)
        if kind == "ballast":
            tokens(ax, pts, m_ballast)
            mark_com(ax, com_u, filled=False)
            label(ax, "uniform SFT", com_u, dx=16, dy=10, ha="left", size=14)
            ax.annotate("", xy=com_b, xytext=com_u, zorder=13,
                        arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.9",
                                        color=COM_BALLAST, lw=4.2, shrinkA=14, shrinkB=14))
            mark_com(ax, com_b, color=COM_BALLAST)
            label(ax, "centre of mass", com_b, dy=-34, color=COM_BALLAST)
            return im
        if kind == "rl":
            tokens(ax, pts_rl, np.ones(len(pts_rl)))
            mark_com(ax, com_rl)
            label(ax, "centre of mass", com_rl, dy=-34)
            return im
        tokens(ax, pts, m_uniform)
        mark_com(ax, com_u)
        if kind == "sft":
            label(ax, "centre of mass", com_u, dy=-34)
            return im
        # Offset right: the KL-penalty caption occupies the space to the left of the marker.
        label(ax, "centre of mass\n(unchanged)", com_u, dx=15, dy=15, ha="left")
        ax.annotate("", xy=tip, xytext=com_u, zorder=13,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.9",
                                    color="#ff3b30", lw=4.2, shrinkA=13, shrinkB=0))
        ax.scatter(*tip, marker="X", s=300, color="#ff3b30", edgecolor="white",
                   linewidths=1.6, zorder=14)
        label(ax, "KL penalty", tip, dy=58, color="#ff6b60", z=14)
        label(ax, "no data here", tip, dy=32, size=14, z=14)
        return im

    def colorbar(fig, im, ax, fraction=0.045):
        cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=0.02)
        cb.set_label(r"KL divergence from the base model,  $\mathrm{KL}(\pi_0\|\pi)$",
                     fontsize=14)
        cb.set_ticks([1, 2, 3, 4, 5])
        cb.ax.tick_params(labelsize=12)

    def save(fig, fname):
        path = os.path.join(args.out_dir, fname)
        fig.savefig(path, dpi=170, bbox_inches="tight")
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}  (+ .pdf)")

    (x0, x1), (y0, y1) = limits(np)
    ratio = (x1 - x0) / (y1 - y0)
    scenes = [("sft", "Standard SFT", "ballast_concept_1_sft.png"),
              ("klpen", "SFT with a KL penalty", "ballast_concept_2_klpenalty.png"),
              ("ballast", f"{args.name} SFT", "ballast_concept_3_ballast.png"),
              ("rl", "RL (on-policy)", "ballast_concept_4_rl.png")]

    if not args.merged_only:
        for kind, title, fname in scenes:
            fig, ax = plt.subplots(figsize=(7.0 * ratio + 1.7, 7.0))
            im = panel(ax, kind, region_label=args.region_label)
            ax.set_title(title, fontsize=17)
            colorbar(fig, im, ax)
            fig.tight_layout()
            save(fig, fname)

    # Merged 2x2: one shared colour scale, region named once on panel (a).
    fig, axes = plt.subplots(2, 2, figsize=(2 * 6.6 * ratio + 2.0, 2 * 6.4))
    for i, (ax, (kind, title, _)) in enumerate(zip(axes.ravel(), scenes)):
        im = panel(ax, kind, region_label=args.region_label if i == 0 else None)
        ax.set_title(f"({'abcd'[i]})  {title}", fontsize=17)
    colorbar(fig, im, list(axes.ravel()), fraction=0.025)
    fig.subplots_adjust(wspace=0.08, hspace=0.18)
    save(fig, "ballast_concept.png")

    # Teaser: just the contrast that carries the idea, stacked so it stays legible at the
    # width of a single ACL column. The 2x2 above is the version for the body of the paper.
    teaser = [("sft", "(a)  Standard SFT"), ("ballast", f"(b)  {args.name} SFT")]
    fig, axes = plt.subplots(2, 1, figsize=(6.6 * ratio, 2 * 6.0))
    for i, (ax, (kind, title)) in enumerate(zip(axes.ravel(), teaser)):
        im = panel(ax, kind, region_label=args.region_label if i == 0 else None)
        ax.set_title(title, fontsize=19)
    colorbar(fig, im, list(axes.ravel()), fraction=0.045)
    fig.subplots_adjust(hspace=0.12)
    save(fig, "ballast_concept_teaser.png")

    print(f"\ncentre-of-mass KL   standard SFT {kl_u:.2f}   {args.name} "
          f"(T={args.temperature:g}) {kl_b:.2f}   ({100 * (kl_u - kl_b) / kl_u:.0f}% lower)"
          f"   RL {kl_rl:.2f}")
    print(f"weights: min {m_ballast.min():.2f}x, max {m_ballast.max():.2f}x the uniform "
          f"token, mean {m_ballast.mean():.2f} (mass conserved)")
    print(f"centre of mass inside the good region?  uniform {inside(np, com_u)}, "
          f"{args.name} {inside(np, com_b)}, RL {inside(np, com_rl)}   |   "
          f"KL-penalty solution inside? {inside(np, tuple(tip))}")


if __name__ == "__main__":
    main()
