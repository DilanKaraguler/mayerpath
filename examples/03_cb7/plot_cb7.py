"""Figures for the CB7 example: Betti numbers and smallest nonzero eigenvalues.

    python plot_cb7.py         Euclidean filtration (results/cb7_curves.csv from run_cb7.py)
                               -> figures/cb7_betti.{pdf,png}, cb7_spectra.{pdf,png}
    python plot_cb7.py --fri   FRI filtration (results/cb7_fri_curves.csv from run_cb7_fri.py)
                               -> figures/cb7_fri_betti.{pdf,png}, cb7_fri_spectra.{pdf,png}

Three full-width panels (n = 0, 1, 2) per figure, N = 2 vs N = 3 (q = 1, 2).
"""
import sys
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = [  # (N, q, label, color, linestyle)
    (2, 1, r"$N=2$ (path homology)", "#2a78d6", "-"),
    (3, 1, r"$N=3,\ q=1$", "#eb6834", "--"),
    (3, 2, r"$N=3,\ q=2$", "#1baf7a", ":"),
]
# shaded filtration values: (low, high, label, text alignment, text x)
EUCLID = dict(csv="cb7_curves.csv", x="r", xlim=(0.5, 1.8), prefix="cb7",
              xlabel=r"filtration radius $r$ (Å)",
              events=[(1.055, 1.095, r"H$\to$N", "right", 1.045), (1.175, 1.185, r"N$\to$O", "left", 1.195)])
FRI = dict(csv="cb7_fri_curves.csv", x="s", xlim=(0.35, 0.95), prefix="cb7_fri",
           xlabel=r"normalized distance $s=r_{ij}/(r_i+r_j)$  (FRI: $d_{ij}=1-e^{-(s/\tau)^\kappa}$)",
           events=[(0.7425, 0.7525, r"C$\to$N", "right", 0.738), (0.7675, 0.7925, r"H$\to$N, N$\to$O", "left", 0.797)])
EVENTS = EUCLID["events"]
FIGURES = [  # (csv field, y label, file name)
    ("betti", r"$\beta_{%d}^{N,q}$", "betti"),
    ("lam", r"$\tilde\lambda_{%d}^{N,q}$", "spectra"),
]
FIGSIZE = (7.0, 6.4)


def load(cfg=EUCLID):
    data = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(HERE, "results", cfg["csv"])) as fh:
        for row in csv.DictReader(fh):
            key = (int(row["N"]), int(row["q"]), int(row["n"]))
            data[key]["r"].append(float(row[cfg["x"]]))
            data[key]["betti"].append(int(row["betti"]))
            data[key]["lam"].append(float(row["lambda_min"]))
    return data


def plot(data, field, ylabel, name, out, cfg=EUCLID):
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    for n, ax in enumerate(axes):
        for lo, hi, *_ in cfg["events"]:
            ax.axvspan(lo, hi, color="0.88", lw=0, zorder=0)
        for N, q, label, color, ls in SERIES:
            d = data[(N, q, n)]
            ax.step(d["r"], d[field], where="post", color=color, ls=ls, lw=1.6, label=label)
        ax.set_ylabel(ylabel % n)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_xlim(*cfg["xlim"])
    for lo, hi, text, ha, x in cfg["events"]:
        axes[0].text(x, 1.02, text, transform=axes[0].get_xaxis_transform(),
                     ha=ha, va="bottom", fontsize=7.5, color="0.35")
    axes[-1].set_xlabel(cfg["xlabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{cfg['prefix']}_{name}.{ext}"), dpi=200)
    plt.close(fig)


def main():
    cfg = FRI if "--fri" in sys.argv[1:] else EUCLID
    data = load(cfg)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for field, ylabel, name in FIGURES:
        plot(data, field, ylabel, name, out, cfg)
    print("wrote", out)


if __name__ == "__main__":
    main()
