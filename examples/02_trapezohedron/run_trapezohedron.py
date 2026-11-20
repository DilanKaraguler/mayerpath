"""Filtered trapezohedron T_3 with two added edges (paper, Section "Example: a
trapezohedron with added edges").

G_0: the eight vertices; G_1: the trapezohedron T_3; G_2 = G_1 + (a -> j);
G_3 = G_2 + (a -> b).  Writes results/trapezohedron.md and
figures/trapezohedron_barcodes.{pdf,png}.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mayerpath import Filtration  # noqa: E402

V = ["a", "i", "i1", "i2", "j", "j1", "j2", "b"]
G1 = [("a", "i"), ("a", "i1"), ("a", "i2"),
      ("i", "j"), ("i", "j1"), ("i1", "j1"), ("i1", "j2"), ("i2", "j2"), ("i2", "j"),
      ("j", "b"), ("j1", "b"), ("j2", "b")]
G2 = G1 + [("a", "j")]
G3 = G2 + [("a", "b")]
STEPS = [(V, []), (V, G1), (V, G2), (V, G3)]
CASES = [(2, 1), (3, 1), (3, 2)]


def main():
    lines = ["# Trapezohedron filtration", "", "## dim Omega_n", "",
             "| N | step | Omega_0 | Omega_1 | Omega_2 | Omega_3 |", "|---|---|---|---|---|---|"]
    filt = {N: Filtration(STEPS, N, max_degree=3) for N in (2, 3)}
    for N, F in filt.items():
        for k, C in enumerate(F.complexes):
            lines.append(f"| {N} | {k} | " + " | ".join(str(C.omega_dim(n)) for n in range(4)) + " |")
    lines += ["", "## Betti numbers at each step", "", "| invariant | step 0 | step 1 | step 2 | step 3 |",
              "|---|---|---|---|---|"]
    for N, q in CASES:
        for n in range(4):
            lines.append(f"| beta_{n}^{{{N},{q}}} | " + " | ".join(map(str, filt[N].betti_curve(n, q))) + " |")
    lines += ["", "## Persistent Betti numbers beta_n^{N,q}(1, 3)", ""]
    for N, q in CASES:
        lines.append(f"- N={N}, q={q}: " + ", ".join(f"n={n}: {filt[N].betti(n, q, 1, 3)}" for n in range(4)))
    lines += ["", "## Barcodes (birth step, death step; None = alive at step 3)", ""]
    bars = {}
    for N, q in CASES:
        for n in range(3):
            bars[(N, q, n)] = filt[N].barcode(n, q)
            lines.append(f"- H_{n}^{{{N},{q}}}: {bars[(N, q, n)]}")
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "trapezohedron.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # barcode figure
    colors = {(2, 1): "#2a78d6", (3, 1): "#eb6834", (3, 2): "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2), sharey=False)
    for ax, (N, q) in zip(axes, CASES):
        y = 0
        ticks, labels = [], []
        for n in range(3):
            start = y
            for b, d in sorted(bars[(N, q, n)], key=lambda t: (t[0], 9 if t[1] is None else t[1])):
                end = 3.6 if d is None else d
                ax.plot([b, end], [y, y], color=colors[(N, q)], lw=3, solid_capstyle="butt")
                if d is None:
                    ax.annotate("", xy=(3.75, y), xytext=(3.55, y),
                                arrowprops=dict(arrowstyle="->", color=colors[(N, q)], lw=1.2))
                y += 1
            if y > start:
                ticks.append((start + y - 1) / 2)
                labels.append(f"$H_{n}$")
                ax.axhline(y - 0.5, color="0.85", lw=0.6)
            y += 0.5
        ax.set_yticks(ticks, labels)
        ax.set_xticks(range(4))
        ax.set_xlim(-0.2, 3.9)
        ax.set_ylim(-0.8, y)
        ax.invert_yaxis()
        ax.set_xlabel("filtration step")
        ax.set_title("path homology ($N=2$)" if N == 2 else f"$N={N},\\ q={q}$", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    figdir = os.path.join(HERE, "figures")
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"trapezohedron_barcodes.{ext}"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
