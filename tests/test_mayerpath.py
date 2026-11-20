"""Tests against the definitions and the worked examples of the paper."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mayerpath import Filtration, MayerPathComplex, harmonic_count, persistent_spectrum  # noqa: E402
from mayerpath.core import allowed_paths, boundary_power, root_of_unity  # noqa: E402
from mayerpath.molecules import electronegativity_digraph, read_mol2  # noqa: E402


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_d_to_the_N_is_zero(N):
    xi = root_of_unity(N)
    for p in [(0, 1, 2, 3, 4, 5), (3, 1, 4, 1, 5, 9, 2)]:
        assert boundary_power(p, xi, N) == {}


def test_omega_is_closed_under_boundary():
    E = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (0, 3), (3, 4), (2, 4)]
    for N in (2, 3, 4):
        C = MayerPathComplex(range(5), E, N, max_degree=2)
        for n in range(1, 4):
            # d of an Omega_n element lies in Omega_{n-1}: B is exact in the Omega bases
            from mayerpath.core import boundary_matrix
            D = boundary_matrix(C.P, n, N, 1) @ C.Q[n]
            if D.shape[1] and C.omega_dim(n - 1):
                Q = C.Q[n - 1].toarray()
                Dd = D.toarray()
                assert np.allclose(Q @ (Q.conj().T @ Dd), Dd)


MOTIFS = {  # Table "Comparison of path and Mayer path homologies for L1, L2, L3"
    "L1": ([(0, 1), (1, 2), (2, 3), (3, 0)], [1, 1], [4, 0], [0, 4]),
    "L2": ([(0, 1), (1, 2), (0, 3), (3, 2)], [1, 0], [3, 1], [1, 3]),
    "L3": ([(0, 2), (0, 3), (1, 2), (1, 3)], [1, 1], [4, 1], [1, 4]),
}


@pytest.mark.parametrize("name", sorted(MOTIFS))
def test_motif_table(name):
    E, b2, b31, b32 = MOTIFS[name]
    assert MayerPathComplex(range(4), E, 2, 1).betti_numbers(1) == b2
    C = MayerPathComplex(range(4), E, 3, 1)
    assert C.betti_numbers(1) == b31
    assert C.betti_numbers(2) == b32


def test_laplacian_examples():
    # 3-cycle, N = 3: L_{0,2} = L_{1,1} has spectrum {1, 1, 4}
    C = MayerPathComplex(range(3), [(0, 1), (1, 2), (2, 0)], 3, 2)
    assert np.allclose(C.spectrum(0, 2), [1, 1, 4])
    assert np.allclose(C.spectrum(1, 1), [1, 1, 4])
    # 3-cycle, N = 2: path Laplacian spectrum {0, 3, 3}
    C2 = MayerPathComplex(range(3), [(0, 1), (1, 2), (2, 0)], 2, 2)
    assert np.allclose(C2.spectrum(0, 1), [0, 3, 3])
    # triangle 0->1->2, 0->2, N = 3
    T = MayerPathComplex(range(3), [(0, 1), (1, 2), (0, 2)], 3, 2)
    expected = sorted([2 * (1 + np.sin(np.pi / 18)), 2 - 2 * np.cos(np.pi / 9), 4 * np.cos(np.pi / 9) ** 2])
    assert np.allclose(T.spectrum(0, 2), expected)
    assert np.allclose(T.spectrum(1, 1), expected)
    assert np.allclose(T.spectrum(0, 1), [0, 0, 3])
    assert np.allclose(T.spectrum(1, 2), [0, 0, 3])
    assert np.allclose(T.spectrum(2, 1), [3])
    assert np.allclose(T.spectrum(2, 2), [3])


def test_example_5_1_same_betti_different_spectra():
    """Example 5.1 (ex:lapvshom) of the paper."""
    GA = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 2)]
    GB = [(0, 2), (0, 3), (1, 0), (1, 2), (3, 2)]
    for N, q in [(2, 1), (3, 1), (3, 2)]:
        a = MayerPathComplex(range(4), GA, N, 3).betti_numbers(q)[:3]
        b = MayerPathComplex(range(4), GB, N, 3).betti_numbers(q)[:3]
        assert a == b
    A3, B3 = MayerPathComplex(range(4), GA, 3, 3), MayerPathComplex(range(4), GB, 3, 3)
    assert np.allclose(A3.spectrum(0, 1), [0, 0, 1, 5])
    assert np.allclose(B3.spectrum(0, 1), [0, 0, 2, 4])
    s2, s3 = np.sqrt(2), np.sqrt(3)
    assert np.allclose(B3.spectrum(0, 2), sorted([2 - s3, 3 - s2, 3 + s2, 2 + s3]))
    for N in (3, 4, 5, 6):
        c = 2 * np.cos(np.pi / N)
        assert np.allclose(MayerPathComplex(range(4), GA, N, 3).spectrum(2, 2), c * c * np.array([1, 5]))
        assert np.allclose(MayerPathComplex(range(4), GB, N, 3).spectrum(2, 2), c * c * np.array([3 - c, 3 + c]))


def random_digraph(rng, n_v, p, acyclic):
    pairs = [(a, b) for a in range(n_v) for b in range(n_v) if a != b and (a < b or not acyclic)]
    return [e for e in pairs if rng.random() < p]


def test_kernel_of_laplacian_is_homology():
    rng = np.random.default_rng(0)
    for trial in range(20):
        E = random_digraph(rng, 7, 0.3, acyclic=trial % 2 == 0)
        for N in (2, 3, 4):
            C = MayerPathComplex(range(7), E, N, 2)
            for q in range(1, N):
                for n in range(3):
                    if n + N - q > 3 and C.P.get(n + N - q):
                        continue                  # would need Omega_4
                    assert harmonic_count(C.spectrum(n, q)) == C.betti(n, q)


def omega_reference_dim(P, n, N):
    """dim Omega_n^N straight from the definition (dense null space), for testing."""
    from mayerpath.core import _null_space
    An = P.get(n, [])
    if not An:
        return 0
    xi = root_of_unity(N)
    rows = {}
    for j, p in enumerate(An):
        for q in range(1, N):
            if n - q < 0:
                break
            for f, c in boundary_power(p, xi, q).items():
                if f not in set(P.get(n - q, [])):
                    rows.setdefault((q, f), np.zeros(len(An), complex))[j] += c
    M = np.array(list(rows.values())) if rows else np.zeros((0, len(An)))
    return _null_space(M).shape[1]


@pytest.mark.parametrize("acyclic", [True, False])
def test_explicit_omega_matches_definition(acyclic):
    """Omega_2 (triangles / zero-sum clusters) and Omega_3 (clusters + coefficient-sum
    condition for N >= 3) agree with the definition, and d^q of the basis stays allowed."""
    from mayerpath.core import boundary_matrix
    rng = np.random.default_rng(1 if acyclic else 2)
    for _ in range(30):
        E = random_digraph(rng, 8, 0.35, acyclic)
        for N in (2, 3, 4, 5):
            C = MayerPathComplex(range(8), E, N, 3)
            for n in (2, 3):
                assert C.omega_dim(n) == omega_reference_dim(C.P, n, N)
                Q = C.Q[n]
                if Q.shape[1]:
                    assert np.allclose((Q.conj().T @ Q).toarray(), np.eye(Q.shape[1]))
                    for q in range(1, min(N, n + 1)):
                        # d^q of Omega_n has no component outside A_{n-q}
                        from mayerpath.core import boundary_power
                        xi = root_of_unity(N)
                        allowed = set(C.P.get(n - q, []))
                        for k in range(Q.shape[1]):
                            col = Q[:, k].toarray().ravel()
                            out = {}
                            for i, c in enumerate(col):
                                if abs(c) > 1e-13:
                                    for f, w in boundary_power(C.P[n][i], xi, q).items():
                                        out[f] = out.get(f, 0) + c * w
                            assert all(abs(v) < 1e-9 for f, v in out.items() if f not in allowed)


def test_two_gamma8_clusters():
    """a->x->y->b and a->x'->y'->b with a->y, x->b, a->y', x'->b and a -/-> b:
    neither 3-path is in Omega_3^N for N >= 3, their difference is."""
    E = [("a", "x"), ("x", "y"), ("y", "b"), ("a", "y"), ("x", "b"),
         ("a", "u"), ("u", "w"), ("w", "b"), ("a", "w"), ("u", "b")]
    V = ["a", "x", "y", "u", "w", "b"]
    assert MayerPathComplex(V, E, 2, 3).omega_dim(3) == 2
    for N in (3, 4, 5):
        assert MayerPathComplex(V, E, N, 3).omega_dim(3) == 1
    single = E[:5]
    assert MayerPathComplex(V[:3] + ["b"], single, 3, 3).omega_dim(3) == 0


def trapezohedron():
    V = ["a", "i", "i1", "i2", "j", "j1", "j2", "b"]
    G1 = [("a", "i"), ("a", "i1"), ("a", "i2"), ("i", "j"), ("i", "j1"), ("i1", "j1"), ("i1", "j2"),
          ("i2", "j2"), ("i2", "j"), ("j", "b"), ("j1", "b"), ("j2", "b")]
    G2 = G1 + [("a", "j")]
    G3 = G2 + [("a", "b")]
    return [(V, []), (V, G1), (V, G2), (V, G3)]


def test_trapezohedron_example():
    F2 = Filtration(trapezohedron(), 2, 3)
    assert F2.betti_curve(0, 1) == [8, 1, 1, 1]
    assert all(F2.betti_curve(n, 1) == [0, 0, 0, 0] for n in (1, 2, 3))
    F = Filtration(trapezohedron(), 3, 3)
    assert F.betti_curve(0, 1) == [8, 4, 3, 2]
    assert F.betti_curve(1, 1) == [0, 4, 4, 5]
    assert F.betti_curve(0, 2) == [8, 1, 0, 0]
    assert F.betti_curve(1, 2) == [0, 7, 7, 7]
    assert F.betti_curve(2, 1)[1:] == [1, 1, 1] and F.betti_curve(2, 2)[1:] == [1, 1, 1]
    assert F.betti(0, 1, 1, 3) == 2
    assert F.betti(1, 1, 1, 3) == 4
    dims = [F.complexes[k].omega_dim(n) for k in (1, 2, 3) for n in (1, 2, 3)]
    assert dims == [12, 6, 1, 13, 7, 1, 14, 8, 1]


def test_persistent_laplacian_kernel():
    F = Filtration(trapezohedron(), 3, 3)
    for n in range(3):
        for q in (1, 2):
            for s in range(1, 4):
                k = harmonic_count(persistent_spectrum(F.complexes[1], F.complexes[s], n, q))
                assert k == F.betti(n, q, 1, s)


def test_cb7_reproduces_wang_wei():
    el, X = read_mol2(os.path.join(ROOT, "data", "cb7.mol2"))
    assert len(el) == 126
    betti0 = {}
    for r in (0.565, 0.710, 0.800):
        V, E = electronegativity_digraph(el, X, 2 * r)
        betti0[r] = MayerPathComplex(V, E, 2, 0).betti(0, 1)
    assert betti0 == {0.565: 28, 0.710: 42, 0.800: 1}
    V, E = electronegativity_digraph(el, X, 2 * 1.68)
    assert MayerPathComplex(V, E, 2, 2).betti(2, 1) == 85


def test_paths_are_allowed():
    P = allowed_paths([0, 1, 2], [(0, 1), (1, 2)], 3)
    assert P[2] == [(0, 1, 2)] and P[3] == []


def test_n3_formulas_match_ranks():
    """Proposition rank-d2 and Corollary betti-N3 (Section "Computing Mayer Path
    Homology when N = 3") agree with the rank computation, also with self-loops
    and opposite edges."""
    from mayerpath import formulas
    rng = np.random.default_rng(3)
    for trial in range(60):
        n_v = 7
        pairs = [(a, b) for a in range(n_v) for b in range(n_v)
                 if (a < b) or (trial % 3 and a != b) or (trial % 3 == 2 and a == b)]
        E = [e for e in pairs if rng.random() < 0.3]
        C = MayerPathComplex(range(n_v), E, 3, 2)
        if any(C.P.get(m) for m in (4, 5)) and trial % 3:
            pass                                    # beta_2^{3,2} needs only Omega_3
        assert formulas.omega2_dim(range(n_v), E) == C.omega_dim(2)
        assert formulas.rank_d2_on_omega2(range(n_v), E) == C.omega_dim(0) - C.betti(0, 1, method="rank")
        assert C.betti(0, 1) == C.betti(0, 1, method="rank")
        assert C.betti(2, 2) == C.betti(2, 2, method="rank")


def test_n3_formula_without_triangles():
    """No triangles: T = 0 and beta_0^{3,1} = |C| (number of square classes)."""
    from mayerpath import formulas
    V = ["a", "i", "i1", "i2", "j", "j1", "j2", "b"]
    G1 = trapezohedron()[1][1]
    assert formulas.triangles(G1) == []
    assert len(set(formulas.square_classes(V, G1).values())) == 4      # [a], [L1], [L2], [b]
    assert MayerPathComplex(V, G1, 3, 2).betti(0, 1) == 4
