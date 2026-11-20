"""Persistent Mayer path homology and Laplacians of cucurbit[7]uril (CB7).

Digraph construction as in Wang & Wei, "Persistent path Laplacian" (2022), Sec. 4:
atoms are vertices; two atoms of different elements at distance <= 2r are joined
by an edge directed by Pauling electronegativity (H -> C -> N -> O); isolated
atoms are removed.  For every radius r on the grid we compute, for N = 2 (q = 1)
and N = 3 (q = 1, 2) and n = 0, 1, 2,

    beta_n^{N,q}(r)      Mayer path Betti number,
    lambda_n^{N,q}(r)    smallest nonzero eigenvalue of the Mayer path Laplacian L_{n,q},

and the persistent Betti numbers between consecutive radii of the eight
filtration values used by Wang & Wei.

Usage:  python examples/03_cb7/run_cb7.py   (writes examples/03_cb7/results/*.csv)
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from mayerpath import Filtration, MayerPathComplex, smallest_nonzero  # noqa: E402
from mayerpath.molecules import electronegativity_digraph, read_mol2   # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "cb7.mol2")
OUT = os.path.join(HERE, "results")
GRID = np.round(np.arange(0.50, 1.8001, 0.01), 3)
WANG_WEI = [0.200, 0.565, 0.710, 0.745, 0.800, 1.210, 1.315, 1.800]
CASES = [(2, 1), (3, 1), (3, 2)]
DEGREES = (0, 1, 2)


def digraph(elements, coords, r):
    return electronegativity_digraph(elements, coords, 2 * r, same_element="drop")


def main():
    os.makedirs(OUT, exist_ok=True)
    elements, coords = read_mol2(DATA)
    t0 = time.time()
    rows = []
    for r in GRID:
        V, E = digraph(elements, coords, r)
        for N in (2, 3):
            C = MayerPathComplex(V, E, N, max_degree=2)
            for q in range(1, N):
                for n in DEGREES:
                    rows.append(dict(r=r, N=N, q=q, n=n, vertices=len(V), edges=len(E),
                                     omega_dim=C.omega_dim(n), betti=C.betti(n, q),
                                     lambda_min=smallest_nonzero(C.spectrum(n, q))))
        print(f"r = {r:.2f}  |V| = {len(V)}  |E| = {len(E)}  ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "cb7_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # persistent Betti numbers on the eight radii of Wang & Wei
    prow = []
    steps = [digraph(elements, coords, r) for r in WANG_WEI]
    for N in (2, 3):
        F = Filtration(steps, N, max_degree=2, values=WANG_WEI)
        for q in range(1, N):
            for n in DEGREES:
                for a in range(len(F)):
                    for b in range(a, len(F)):
                        prow.append(dict(N=N, q=q, n=n, r_t=WANG_WEI[a], r_s=WANG_WEI[b],
                                         betti=F.betti(n, q, a, b)))
    with open(os.path.join(OUT, "cb7_persistent_betti.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
