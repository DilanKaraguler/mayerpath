"""Directed graphs from molecular structures."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Bondi van der Waals radii (Angstrom)
VDW_RADIUS = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "Cl": 1.75}

# Pauling electronegativities
PAULING = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "P": 2.19, "F": 3.98, "Cl": 3.16}


def read_mol2(path: str) -> Tuple[List[str], np.ndarray]:
    """Element symbols and coordinates (Angstrom) from the ATOM block of a .mol2 file."""
    elements, coords, inside = [], [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>"):
                inside = line.strip() == "@<TRIPOS>ATOM"
                continue
            if inside and line.strip():
                t = line.split()
                sybyl = t[5]                      # e.g. C.ar, N.am, O.2, H
                elements.append("".join(ch for ch in t[1] if ch.isalpha()).capitalize())
                coords.append([float(x) for x in t[2:5]])
    return elements, np.array(coords)


def fri_distance(elements: Sequence[str], coords: np.ndarray, tau: float = 1.0, kappa: float = 2.0):
    """FRI distance d_ij = 1 - exp(-(r_ij / eta_ij)^kappa), eta_ij = tau (r_i + r_j),
    with r_i the van der Waals radius (Xia, Opron and Wei).

    d_ij is an increasing function of the normalized distance r_ij / (r_i + r_j),
    so tau and kappa only reparametrize the filtration: the digraphs, and hence
    all (persistent) Betti numbers, depend only on the order of r_ij / (r_i + r_j).
    """
    R = np.array([VDW_RADIUS[e] for e in elements])
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    eta = tau * (R[:, None] + R[None, :])
    return 1.0 - np.exp(-(D / eta) ** kappa)


def electronegativity_digraph(elements: Sequence[str], coords: np.ndarray, cutoff: float,
                              same_element: str = "drop", keep_isolated: bool = False,
                              distance: np.ndarray | None = None):
    """Digraph on atoms with an edge for every pair at distance <= cutoff.

    Edges point from the less to the more electronegative atom (H -> C -> N -> O),
    as in Wang and Wei, "Persistent path Laplacian" (2022).  Pairs of atoms of the
    same element have no direction under this rule:

    * same_element="drop"  (default) leaves them out, which reproduces the
      Betti numbers reported by Wang and Wei for cucurbit[7]uril;
    * same_element="index" directs them from the lower to the higher atom index.

    Either way the digraph is acyclic.  Isolated atoms are dropped unless
    keep_isolated is True.  ``distance`` replaces the Euclidean distance matrix
    (e.g. by fri_distance); the edge criterion is then distance <= cutoff.
    """
    if same_element not in ("drop", "index"):
        raise ValueError("same_element must be 'drop' or 'index'")
    en = [PAULING[e] for e in elements]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) if distance is None else distance
    i, j = np.triu_indices(len(elements), 1)
    mask = D[i, j] <= cutoff + 1e-9
    edges = []
    for a, b in zip(i[mask], j[mask]):
        a, b = int(a), int(b)
        if en[a] == en[b] and same_element == "drop":
            continue
        edges.append((a, b) if (en[a], a) < (en[b], b) else (b, a))
    verts = set(range(len(elements))) if keep_isolated else {v for e in edges for v in e}
    return sorted(verts), edges
