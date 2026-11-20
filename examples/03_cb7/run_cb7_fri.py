"""CB7 with the FRI distance instead of the Euclidean distance.

d_ij = 1 - exp(-(r_ij / eta_ij)^kappa), eta_ij = tau (r_i + r_j), r_i the van der
Waals radius.  Since d_ij increases with s_ij = r_ij / (r_i + r_j), the filtration
is indexed by s (tau and kappa only relabel the filtration values).  Same edge
rule as run_cb7.py: different elements, directed H -> C -> N -> O.

Writes results/cb7_fri_curves.csv, results/cb7_fri_events.md and
results/cb7_fri_persistent_betti.csv (persistent Betti numbers between the
stages bonded framework / C->N contacts / H->N, N->O contacts / end).
"""
import csv
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from mayerpath import Filtration, MayerPathComplex, smallest_nonzero  # noqa: E402
from mayerpath.molecules import VDW_RADIUS, electronegativity_digraph, read_mol2  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "cb7.mol2")
GRID = np.round(np.arange(0.30, 0.9501, 0.005), 4)       # normalized distance s
STAGES = [0.46, 0.75, 0.79, 0.95]


def main():
    el, X = read_mol2(DATA)
    R = np.array([VDW_RADIUS[e] for e in el])
    S = np.linalg.norm(X[:, None] - X[None], axis=2) / (R[:, None] + R[None, :])
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    rows, prev, events, t0 = [], set(), [], time.time()
    for s in GRID:
        V, E = electronegativity_digraph(el, X, s, distance=S)
        new = set(E) - prev
        if new:
            events.append((s, Counter(f"{el[a]}->{el[b]}" for a, b in new)))
        prev = set(E)
        for N in (2, 3):
            C = MayerPathComplex(V, E, N, max_degree=2)
            for q in range(1, N):
                for n in (0, 1, 2):
                    rows.append(dict(s=s, N=N, q=q, n=n, vertices=len(V), edges=len(E),
                                     betti=C.betti(n, q), lambda_min=smallest_nonzero(C.spectrum(n, q))))
    with open(os.path.join(out, "cb7_fri_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "cb7_fri_events.md"), "w") as fh:
        fh.write("# Edges entering along the FRI filtration (s = r_ij / (r_i + r_j))\n\n| s | new edges |\n|---|---|\n")
        for s, c in events:
            fh.write(f"| {s:.3f} | " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) + " |\n")
    steps = [electronegativity_digraph(el, X, s, distance=S) for s in STAGES]
    prow = []
    for N in (2, 3):
        F = Filtration(steps, N, max_degree=2, values=STAGES)
        for q in range(1, N):
            for n in (0, 1, 2):
                for a in range(len(F)):
                    for b in range(a, len(F)):
                        prow.append(dict(N=N, q=q, n=n, s_u=STAGES[a], s_v=STAGES[b], betti=F.betti(n, q, a, b)))
    with open(os.path.join(out, "cb7_fri_persistent_betti.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
