"""Persistent Mayer path homology and persistent Mayer path Laplacians.

For a filtration G_t subset G_s and 1 <= q <= N-1, with m = n + N - q:

    Omega_m(t,s)       = {v in Omega_m(G_s) : d^{N-q} v in Omega_n(G_t)}
    L^{t,s}_{n,q}      = (d_t^q)^* d_t^q + d^{N-q}_{t,s} (d^{N-q}_{t,s})^*  on Omega_n(G_t)
    beta_n^{N,q}(t,s)  = rank(H_n^{N,q}(G_t) -> H_n^{N,q}(G_s)) = dim ker L^{t,s}_{n,q}

The up part uses an orthonormal basis of Omega_m(t,s), as in the paper.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .complex import MayerPathComplex, _clean, harmonic_count, smallest_nonzero
from .core import boundary_matrix, rank, TOL


def _embed(Ct: MayerPathComplex, Cs: MayerPathComplex, n: int) -> np.ndarray:
    """Orthonormal basis of Omega_n(G_t) in the path coordinates of G_s."""
    idx = {p: i for i, p in enumerate(Cs.P.get(n, []))}
    Qt = Ct.Q[n].tocoo()
    rows = [idx[Ct.P[n][r]] for r in Qt.row]   # raises KeyError if G_t is not a subdigraph
    return sp.csc_matrix((Qt.data, (rows, Qt.col)),
                         shape=(len(Cs.P.get(n, [])), Qt.shape[1])).toarray()


def _check_nested(Ct: MayerPathComplex, Cs: MayerPathComplex):
    if Ct.N != Cs.N:
        raise ValueError("both complexes need the same N")
    if not set(Ct.vertices) <= set(Cs.vertices) or not set(Ct.edges) <= set(Cs.edges):
        raise ValueError("G_t must be a subdigraph of G_s")


def up_map(Ct: MayerPathComplex, Cs: MayerPathComplex, n: int, q: int) -> np.ndarray:
    """Matrix of d^{N-q}_{t,s} : Omega_{n+N-q}(t,s) -> Omega_n(G_t), orthonormal bases."""
    _check_nested(Ct, Cs)
    N, m = Cs.N, n + Cs.N - q
    dn_t = Ct.omega_dim(n)
    if dn_t == 0 or Cs.omega_dim(m) == 0:
        return np.zeros((dn_t, 0), complex)
    R = _embed(Ct, Cs, n)                                   # A_n(s) x dim Omega_n(t)
    W = (boundary_matrix(Cs.P, m, N, N - q) @ Cs.Q[m]).toarray()   # d^{N-q} of Omega_m(s)
    resid = W - R @ (R.conj().T @ W)                        # part outside Omega_n(t)
    if resid.size:
        _, s, vh = np.linalg.svd(resid)
        r = int((s > TOL * max(1.0, s[0] if s.size else 1.0)).sum())
        X = vh[r:].conj().T                                 # orthonormal basis of Omega_m(t,s)
    else:
        X = np.eye(W.shape[1], dtype=complex)
    return R.conj().T @ (W @ X)


def persistent_laplacian(Ct: MayerPathComplex, Cs: MayerPathComplex, n: int, q: int) -> np.ndarray:
    Ct._check_q(q)
    d = Ct.omega_dim(n)
    L = np.zeros((d, d), complex)
    if n - q >= 0:
        Bd = Ct.B(n, q)
        L += Bd.conj().T @ Bd
    Bu = up_map(Ct, Cs, n, q)
    L += Bu @ Bu.conj().T
    return L


def persistent_spectrum(Ct, Cs, n, q) -> np.ndarray:
    L = persistent_laplacian(Ct, Cs, n, q)
    return np.zeros(0) if L.size == 0 else _clean(np.linalg.eigvalsh(L))


def persistent_betti(Ct: MayerPathComplex, Cs: MayerPathComplex, n: int, q: int) -> int:
    """rank(H_n^{N,q}(G_t) -> H_n^{N,q}(G_s)) = dim Z_n(t) - rank of d^{N-q}_{t,s}."""
    Ct._check_q(q)
    down = rank(Ct.B(n, q)) if n - q >= 0 else 0
    return Ct.omega_dim(n) - down - rank(up_map(Ct, Cs, n, q))


class Filtration:
    """A nested sequence of digraphs G_0 subset G_1 subset ... with their complexes."""

    def __init__(self, steps: Sequence[Tuple[Iterable, Iterable]], N: int, max_degree: int = 2,
                 values: Sequence[float] | None = None):
        self.N = N
        self.values = list(values) if values is not None else list(range(len(steps)))
        self.complexes: List[MayerPathComplex] = [MayerPathComplex(V, E, N, max_degree) for V, E in steps]
        for a, b in zip(self.complexes, self.complexes[1:]):
            _check_nested(a, b)

    def __len__(self):
        return len(self.complexes)

    def betti(self, n: int, q: int, a: int, b: int | None = None) -> int:
        b = a if b is None else b
        if b == a:
            return self.complexes[a].betti(n, q)
        return persistent_betti(self.complexes[a], self.complexes[b], n, q)

    def betti_curve(self, n: int, q: int) -> List[int]:
        return [self.complexes[a].betti(n, q) for a in range(len(self))]

    def spectrum(self, n: int, q: int, a: int, b: int | None = None) -> np.ndarray:
        b = a if b is None else b
        if b == a:
            return self.complexes[a].spectrum(n, q)
        return persistent_spectrum(self.complexes[a], self.complexes[b], n, q)

    def smallest_nonzero_curve(self, n: int, q: int) -> List[float]:
        return [smallest_nonzero(self.complexes[a].spectrum(n, q)) for a in range(len(self))]

    def barcode(self, n: int, q: int):
        """Bars (birth, death) of H_n^{N,q}; death = None for classes alive at the last step.

        Uses beta(a,b) for all a <= b and the usual inclusion-exclusion
        (number of bars born at a and dying at b).
        """
        K = len(self)
        beta = np.zeros((K + 1, K + 1), dtype=int)
        for a in range(K):
            for b in range(a, K):
                beta[a, b] = self.betti(n, q, a, b)
        bars = []
        for a in range(K):
            for b in range(a, K):
                # classes born at a (not before) and alive at b but not at b+1
                alive = beta[a, b] - (beta[a - 1, b] if a > 0 else 0)
                nxt = (beta[a, b + 1] - (beta[a - 1, b + 1] if a > 0 else 0)) if b + 1 < K else 0
                for _ in range(alive - nxt):
                    bars.append((self.values[a], self.values[b + 1] if b + 1 < K else None))
        return bars
