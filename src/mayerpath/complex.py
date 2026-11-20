"""Mayer path complex of a digraph, its homology and Laplacians."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp

from .core import MAX_DEGREE, vertex_key, allowed_paths, boundary_matrix, omega_basis, rank, TOL


class MayerPathComplex:
    """Omega_n^N(G) for n <= max_degree (+ the degrees the boundaries need).

    Parameters
    ----------
    vertices, edges : vertex labels and directed edges (a, b), a != b.
    N : Mayer order, N >= 2 (N = 2 is ordinary path homology).
    max_degree : highest degree n for which homology / Laplacians are wanted.
    """

    def __init__(self, vertices: Iterable, edges: Iterable[Tuple], N: int, max_degree: int = 2):
        if N < 2:
            raise ValueError("N must be >= 2")
        if not 0 <= max_degree <= MAX_DEGREE:
            raise ValueError(f"max_degree must be between 0 and {MAX_DEGREE}")
        self.N = N
        self.max_degree = max_degree
        self.vertices = sorted(set(vertices), key=vertex_key)
        self.edges = sorted(set(map(tuple, edges)), key=lambda e: (vertex_key(e[0]), vertex_key(e[1])))   # self-loops allowed
        top = max_degree + N - 1          # Omega_{n+N-q} with q = 1
        self.P = allowed_paths(self.vertices, self.edges, top)
        self.Q: Dict[int, sp.csc_matrix] = {n: omega_basis(self.P, n, N)
                                            for n in range(min(top, MAX_DEGREE) + 1)}
        self._B: Dict[Tuple[int, int], np.ndarray] = {}

    # -- dimensions -------------------------------------------------------
    def omega_dim(self, n: int) -> int:
        """dim Omega_n^N.  For n > 3 this is 0 when there are no allowed n-paths;
        otherwise Omega_n is not constructed and an error is raised."""
        if n < 0:
            return 0
        if n > MAX_DEGREE:
            if self.P.get(n):
                raise NotImplementedError(
                    f"needs Omega_{n}, but Omega_n is only constructed for n <= {MAX_DEGREE} "
                    f"and this digraph has allowed {n}-paths")
            return 0
        Q = self.Q.get(n)
        return 0 if Q is None else Q.shape[1]

    # -- boundary d^k : Omega_n -> Omega_{n-k} in orthonormal Omega bases --
    def B(self, n: int, k: int) -> np.ndarray:
        key = (n, k)
        if key not in self._B:
            dn, dt = self.omega_dim(n), (self.omega_dim(n - k) if n - k >= 0 else 0)
            if dn == 0 or dt == 0:
                M = np.zeros((dt, dn), complex)
            else:
                D = boundary_matrix(self.P, n, self.N, k)
                M = (self.Q[n - k].conj().T @ (D @ self.Q[n])).toarray()
            self._B[key] = M
        return self._B[key]

    # -- homology ---------------------------------------------------------
    def betti(self, n: int, q: int, method: str = "auto") -> int:
        """Mayer path Betti number beta_n^{N,q}.

        method="auto" uses the closed formulas for N = 3 (beta_0^{3,1} and
        beta_2^{3,2}, Corollary in Section "Computing Mayer Path Homology when
        N = 3") and ranks of boundary matrices otherwise; method="rank" always
        uses ranks.
        """
        self._check_q(q)
        if method == "auto" and self.N == 3 and (n, q) in ((0, 1), (2, 2)):
            from . import formulas
            if (n, q) == (0, 1):
                return formulas.betti0_31(self.vertices, self.edges)
            return formulas.betti2_32(self.vertices, self.edges, rank(self.B(3, 1)))
        down = rank(self.B(n, q)) if n - q >= 0 else 0
        up = rank(self.B(n + self.N - q, self.N - q))
        return self.omega_dim(n) - down - up

    def betti_numbers(self, q: int, max_degree: int | None = None) -> List[int]:
        m = self.max_degree if max_degree is None else max_degree
        return [self.betti(n, q) for n in range(m + 1)]

    # -- Laplacian --------------------------------------------------------
    def laplacian(self, n: int, q: int) -> np.ndarray:
        """Matrix of L_{n,q} = (d^q)^* d^q + d^{N-q} (d^{N-q})^* on Omega_n."""
        self._check_q(q)
        d = self.omega_dim(n)
        L = np.zeros((d, d), complex)
        if n - q >= 0:
            Bd = self.B(n, q)
            L += Bd.conj().T @ Bd
        Bu = self.B(n + self.N - q, self.N - q)
        L += Bu @ Bu.conj().T
        return L

    def spectrum(self, n: int, q: int) -> np.ndarray:
        L = self.laplacian(n, q)
        if L.size == 0:
            return np.zeros(0)
        return _clean(np.linalg.eigvalsh(L))

    def _check_q(self, q):
        if not 1 <= q <= self.N - 1:
            raise ValueError(f"q must satisfy 1 <= q <= N-1 = {self.N - 1}")


def _clean(ev: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Clip round-off: eigenvalues below tol * max(1, largest) are set to 0."""
    ev = np.clip(ev, 0, None)
    ev[ev <= tol * max(1.0, float(ev.max(initial=0.0)))] = 0.0
    return ev


def harmonic_count(spec: np.ndarray, tol: float = 1e-8) -> int:
    scale = max(1.0, float(spec.max(initial=0.0)))
    return int((spec <= tol * scale).sum())


def smallest_nonzero(spec: np.ndarray, tol: float = 1e-8) -> float:
    scale = max(1.0, float(spec.max(initial=0.0)))
    nz = spec[spec > tol * scale]
    return float(nz.min()) if nz.size else 0.0
