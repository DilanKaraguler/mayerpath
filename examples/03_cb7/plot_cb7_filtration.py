"""Images of the CB7 digraph filtration.

Atoms are drawn at their coordinates (orthographic projection), coloured by
element; every edge is drawn as an arrow H -> C -> N -> O, and the edges that
entered since the previous panel are highlighted.

    python plot_cb7_filtration.py         Euclidean filtration (radius r, edges at distance <= 2r)
        -> figures/cb7_filtration.{pdf,png}, cb7_filtration_unit.{pdf,png}
    python plot_cb7_filtration.py --fri   FRI filtration (s = r_ij / (r_i + r_j))
        -> figures/cb7_fri_filtration.{pdf,png}, cb7_fri_filtration_unit.{pdf,png}

"..._unit" shows one glycoluril unit with its four methylene bridges (side view),
the other the whole molecule viewed along the ring axis.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from mayerpath.molecules import VDW_RADIUS, electronegativity_digraph, read_mol2  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "cb7.mol2")
EUCLID = dict(prefix="cb7", label="r", unit=" Å",
              stages=[0.565, 0.710, 0.745, 1.09, 1.18, 1.37],
              captions=["C–H bonds", "C=O, first C–N", "bonded framework",
                        r"H$\to$N contacts", r"N$\to$O contacts", r"$\beta_0^{3,1}=1$"])
FRI = dict(prefix="cb7_fri", label="s", unit="",
           stages=[0.39, 0.44, 0.46, 0.75, 0.79, 0.94],
           captions=["C=O, C–H bonds", "first C–N bonds", "bonded framework",
                     r"C$\to$N across the ring", r"H$\to$N, N$\to$O contacts", r"$\beta_0^{3,1}=1$"])
ELEMENT_COLOR = {"H": "#d9d9d9", "C": "#404040", "N": "#2f5fd0", "O": "#d62728"}
ELEMENT_SIZE = {"H": 10, "C": 22, "N": 22, "O": 22}
OLD_EDGE, NEW_EDGE = "0.62", "#eb6834"


def frame(coords):
    """Orthonormal frame: e3 = ring axis (least variance), e1, e2 in the ring plane."""
    X = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[0], vt[1], vt[2]


def draw(ax, P, depth, elements, verts, edges, new_edges, atom_scale, arrow_scale, lw):
    order = sorted(depth, key=depth.get)            # far atoms first
    vset = set(verts)
    for a, b in sorted(edges, key=lambda e: depth[e[0]] + depth[e[1]]):
        new = (a, b) in new_edges
        ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=arrow_scale,
                                     lw=lw * (1.4 if new else 1.0), color=NEW_EDGE if new else OLD_EDGE,
                                     shrinkA=2.5, shrinkB=3.5, zorder=2 if new else 1))
    for i in order:
        alpha = 1.0 if i in vset else 0.25          # isolated atoms (removed from G_r) are faded
        ax.scatter(*P[i], s=ELEMENT_SIZE[elements[i]] * atom_scale, c=ELEMENT_COLOR[elements[i]],
                   edgecolors="0.15", linewidths=0.4, alpha=alpha, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def legend(fig):
    handles = [Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=ELEMENT_COLOR[e],
                      markeredgecolor="0.15", markeredgewidth=0.5, label=e) for e in "HCNO"]
    handles += [Line2D([], [], color=OLD_EDGE, lw=1.2, label="edge of $G_r$"),
                Line2D([], [], color=NEW_EDGE, lw=1.6, label="new since previous panel")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5)


def panels(name, P, depth, elements, coords, mask, atom_scale, arrow_scale, lw, figsize, cfg, dist):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    prev = set()
    for ax, r, cap in zip(axes.ravel(), cfg["stages"], cfg["captions"]):
        if dist is None:
            V, E = electronegativity_digraph(elements, coords, 2 * r)
        else:
            V, E = electronegativity_digraph(elements, coords, r, distance=dist)
        E = set(E)
        Ev = {(a, b) for a, b in E if mask[a] and mask[b]}
        new = Ev - prev
        idx = np.nonzero(mask)[0]
        draw(ax, {i: P[i] for i in idx}, {i: depth[i] for i in idx}, elements,
             [v for v in V if mask[v]], Ev, new, atom_scale, arrow_scale, lw)
        ax.set_title(f"${cfg['label']}={r}${cfg['unit']}  ({cap})", fontsize=9)
        prev = Ev
    legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{cfg['prefix']}_{name}.{ext}"), dpi=220)
    plt.close(fig)


def main():
    cfg = FRI if "--fri" in sys.argv[1:] else EUCLID
    elements, coords = read_mol2(DATA)
    dist = None
    if cfg is FRI:
        R = np.array([VDW_RADIUS[e] for e in elements])
        dist = np.linalg.norm(coords[:, None] - coords[None], axis=2) / (R[:, None] + R[None, :])
    c0 = coords.mean(axis=0)
    e1, e2, e3 = frame(coords)
    X = coords - c0

    # whole molecule, viewed along the ring axis, slightly tilted
    t = np.deg2rad(25)
    up = np.cos(t) * e2 + np.sin(t) * e3
    view = np.cross(e1, up)
    P = np.stack([X @ e1, X @ up], axis=1)
    panels("filtration", P, X @ view, elements, coords, np.ones(len(elements), bool),
           atom_scale=0.55, arrow_scale=4, lw=0.45, figsize=(10, 7.2), cfg=cfg, dist=dist)

    # one glycoluril unit: atoms within one seventh of the ring around a carbonyl O
    phi = np.arctan2(X @ e2, X @ e1)
    o = [i for i, e in enumerate(elements) if e == "O"]
    o_top = max(o, key=lambda i: X[i] @ e3)
    # the unit's two carbonyl O lie at the same azimuth (top and bottom rim)
    dphi = np.angle(np.exp(1j * (phi - phi[o_top])))
    mask = np.abs(dphi) <= np.pi / 7 + 0.12
    radial = np.cos(phi[o_top]) * e1 + np.sin(phi[o_top]) * e2
    side = np.cross(e3, radial)
    P = np.stack([X @ side, X @ e3], axis=1)
    panels("filtration_unit", P, X @ radial, elements, coords, mask,
           atom_scale=4.0, arrow_scale=9, lw=0.9, figsize=(10, 7.6), cfg=cfg, dist=dist)
    print("wrote", os.path.join(HERE, "figures"))


if __name__ == "__main__":
    main()
