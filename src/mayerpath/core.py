"""Core linear algebra for Mayer path homology and Mayer path Laplacians.

Conventions (non-regular boundary, coefficients in C):

    d e_{i0...in} = sum_j xi^j e_{i0..^ij..in},   xi = exp(2 pi i / N)
    A_n           = span{allowed elementary n-paths}
    Omega_n^N     = {v in A_n : d^q v in A_{n-q}, q = 1..N-1}
    H_n^{N,q}     = ker(d^q|Omega_n) / im(d^{N-q}|Omega_{n+N-q})
    L_{n,q}       = (d^q)^* d^q + d^{N-q} (d^{N-q})^*   on Omega_n

Omega_n is constructed explicitly for n <= 3 (all that is needed for the
paper): Omega_0 and Omega_1 are spanned by vertices and edges, Omega_2 by
triangles and zero-sum combinations of paths with common endpoints, and
Omega_3 cluster by cluster (one cluster per endpoint pair), using the
characterization of Omega_3^N for N >= 3.  The bases are orthonormal.
"""
from __future__ import annotations

import cmath
from collections import defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

Path = Tuple[Hashable, ...]
TOL = 1e-8


def root_of_unity(N: int) -> complex:
    return cmath.exp(2j * cmath.pi / N)


def faces(p: Path, xi: complex):
    """(face, coefficient) pairs of the Mayer boundary of one elementary path."""
    if len(p) == 1:
        return []
    return [(p[:j] + p[j + 1:], xi ** j) for j in range(len(p))]


def boundary_power(p: Path, xi: complex, k: int) -> Dict[Path, complex]:
    """d^k e_p as a sparse dict."""
    cur = {p: 1.0 + 0j}
    for _ in range(k):
        nxt: Dict[Path, complex] = defaultdict(complex)
        for path, c in cur.items():
            for f, w in faces(path, xi):
                nxt[f] += c * w
        cur = {f: c for f, c in nxt.items() if abs(c) > 1e-12}
    return cur


def vertex_key(v):
    """Sort key: numbers in numeric order, then everything else by its string."""
    return (0, v, "") if isinstance(v, (int, float, np.integer)) else (1, 0, str(v))


def allowed_paths(vertices: Sequence, edges, max_degree: int) -> Dict[int, List[Path]]:
    """All allowed elementary paths (consecutive vertices joined by an edge)."""
    out = defaultdict(list)
    for a, b in edges:
        out[a].append(b)
    for a in out:
        out[a].sort(key=vertex_key)
    P = {0: [(v,) for v in sorted(vertices, key=vertex_key)]}
    for n in range(1, max_degree + 1):
        P[n] = [p + (w,) for p in P[n - 1] for w in out[p[-1]]]
    return P


MAX_DEGREE = 3        # Omega_n is constructed for n <= 3 only


def _orthonormal_zero_sum(k: int) -> np.ndarray:
    """Orthonormal basis (k x (k-1)) of {c in R^k : c_1 + ... + c_k = 0} (Helmert basis)."""
    H = np.zeros((k, k - 1))
    for m in range(1, k):
        H[:m, m - 1] = 1.0
        H[m, m - 1] = -m
        H[:, m - 1] /= np.sqrt(m * (m + 1))
    return H


def _null_space(M: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Orthonormal basis (columns) of ker M via SVD; M is one small cluster."""
    if M.shape[0] == 0:
        return np.eye(M.shape[1], dtype=complex)
    _, s, vh = np.linalg.svd(M)
    r = int((s > tol * max(1.0, s[0] if s.size else 1.0)).sum())
    return vh[r:].conj().T


def _by_endpoints(paths: List[Path]) -> Dict[Tuple, List[int]]:
    groups: Dict[Tuple, List[int]] = defaultdict(list)
    for j, p in enumerate(paths):
        groups[(p[0], p[-1])].append(j)
    return groups


def _assemble(columns: List[Tuple[List[int], np.ndarray]], n_rows: int) -> sp.csc_matrix:
    rows, cols, vals, k = [], [], [], 0
    for members, K in columns:
        for c in range(K.shape[1]):
            for i, j in enumerate(members):
                if abs(K[i, c]) > 1e-13:
                    rows.append(j); cols.append(k); vals.append(K[i, c])
            k += 1
    return sp.csc_matrix((vals, (rows, cols)), shape=(n_rows, k), dtype=complex)


def omega2_basis(P: Dict[int, List[Path]]) -> sp.csc_matrix:
    """Orthonormal basis of Omega_2^N (the same for every N >= 2).

    For v = sum c_p e_p in A_2, only the middle deletion of e_{a m b} can leave
    A_1 (it gives e_{ab}, coefficient xi), and d^2 always lands in A_0.  Hence,
    grouping the 2-paths by their endpoints (a, b):
      * a -> b : every e_{a m b} is in Omega_2 (triangles);
      * a -/-> b : Omega_2 contains exactly the combinations with sum c_m = 0
        (squares e_{amb} - e_{am'b}; for a = b these are differences of double
        edges e_{a m a} - e_{a m' a}).
    """
    A2 = P.get(2, [])
    allowed1 = set(P.get(1, []))
    cols = []
    for (a, b), members in _by_endpoints(A2).items():
        if (a, b) in allowed1:
            cols.append((members, np.eye(len(members), dtype=complex)))
        elif len(members) >= 2:
            cols.append((members, _orthonormal_zero_sum(len(members)).astype(complex)))
    return _assemble(cols, len(A2))


def omega3_basis(P: Dict[int, List[Path]], N: int) -> sp.csc_matrix:
    """Orthonormal basis of Omega_3^N.

    d^q-invariance for q = 1 only involves the interior deletions of
    e_{a i j b}, which keep the endpoints (a, b); so Omega_3^{N,1} splits into
    independent clusters, one for each endpoint pair (a, b), with the
    constraints  [e_{ajb} not allowed]  and  [e_{aib} not allowed].
    For N >= 3 the extra condition d^2 v in A_1 is, by the characterization of
    Omega_3^N in the paper, one linear condition per cluster with a -/-> b:
    the coefficients of the cluster sum to zero.  The characterization assumes
    3-paths with four distinct vertices; clusters that contain a repeated
    vertex (directed 2- or 3-cycles) are handled by imposing d^2 v in A_1
    directly.
    """
    A3 = P.get(3, [])
    if not A3:
        return sp.csc_matrix((0, 0), dtype=complex)
    xi = root_of_unity(N)
    allowed2, allowed1 = set(P.get(2, [])), set(P.get(1, []))
    groups = _by_endpoints(A3)
    simple = all(len(set(A3[j])) == 4 for members in groups.values() for j in members)
    if N >= 3 and not simple:
        return _omega3_with_cycles(P, N)
    cols = []
    for (a, b), members in groups.items():
        faces_ = sorted({f for j in members for f in _interior_faces(A3[j]) if f not in allowed2})
        fidx = {f: r for r, f in enumerate(faces_)}
        extra = 1 if (N >= 3 and (a, b) not in allowed1) else 0
        M = np.zeros((len(faces_) + extra, len(members)), complex)
        for c, j in enumerate(members):
            i_, j_ = A3[j][1], A3[j][2]
            for f, coef in (((a, j_, b), xi), ((a, i_, b), xi ** 2)):
                if f in fidx:
                    M[fidx[f], c] += coef
            if extra:
                M[-1, c] = 1.0                    # sum of the cluster's coefficients = 0
        K = _null_space(M)
        if K.shape[1]:
            cols.append((members, K))
    return _assemble(cols, len(A3))


def _interior_faces(p: Path):
    return [p[:1] + p[2:], p[:2] + p[3:]]


def _omega3_with_cycles(P: Dict[int, List[Path]], N: int) -> sp.csc_matrix:
    """Omega_3^N when some 3-path repeats a vertex: impose d^q v in A_{3-q},
    q = 1, 2, directly.  Clusters are the connected components of 3-paths that
    share a non-allowed face."""
    A3 = P[3]
    xi = root_of_unity(N)
    allowed = {k: set(P.get(k, [])) for k in (1, 2)}
    cons = []
    for p in A3:
        row = {}
        for q in (1, 2):
            for f, c in boundary_power(p, xi, q).items():
                if f not in allowed[3 - q]:
                    row[(q, f)] = c
        cons.append(row)
    parent = list(range(len(A3)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for j, row in enumerate(cons):
        for key in row:
            if key in owner:
                parent[find(j)] = find(owner[key])
            else:
                owner[key] = j
    comps = defaultdict(list)
    for j in range(len(A3)):
        comps[find(j)].append(j)
    cols = []
    for members in comps.values():
        keys = sorted({k for j in members for k in cons[j]}, key=str)
        kidx = {k: r for r, k in enumerate(keys)}
        M = np.zeros((len(keys), len(members)), complex)
        for c, j in enumerate(members):
            for k, v in cons[j].items():
                M[kidx[k], c] = v
        K = _null_space(M)
        if K.shape[1]:
            cols.append((members, K))
    return _assemble(cols, len(A3))


def omega_basis(P: Dict[int, List[Path]], n: int, N: int) -> sp.csc_matrix:
    """Orthonormal basis of Omega_n^N as a sparse (|A_n| x dim) matrix, n <= 3."""
    An = P.get(n, [])
    if n <= 1:                                     # Omega_0 = A_0, Omega_1 = A_1
        return sp.identity(len(An), dtype=complex, format="csc")
    if n == 2:
        return omega2_basis(P)
    if n == 3:
        return omega3_basis(P, N)
    if An:
        raise NotImplementedError(
            f"Omega_{n} is only constructed for n <= {MAX_DEGREE}; this digraph has allowed {n}-paths")
    return sp.csc_matrix((0, 0), dtype=complex)


def boundary_matrix(P: Dict[int, List[Path]], n: int, N: int, k: int = 1) -> sp.csr_matrix:
    """d^k : A_n -> A_{n-k} in path coordinates (rows: A_{n-k})."""
    src = P.get(n, [])
    tgt = P.get(n - k, []) if n - k >= 0 else []
    if not src or not tgt:
        return sp.csr_matrix((len(tgt), len(src)), dtype=complex)
    xi = root_of_unity(N)
    tidx = {p: i for i, p in enumerate(tgt)}
    rows, cols, vals = [], [], []
    for j, p in enumerate(src):
        for f, c in boundary_power(p, xi, k).items():
            i = tidx.get(f)
            if i is not None:           # faces outside A_{n-k} cancel on Omega
                rows.append(i); cols.append(j); vals.append(c)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(tgt), len(src)), dtype=complex)


def rank(M, tol: float = TOL) -> int:
    """Numerical rank (singular values above tol * largest singular value)."""
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    if M.size == 0 or min(M.shape) == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s > tol * max(1.0, s[0])).sum())
