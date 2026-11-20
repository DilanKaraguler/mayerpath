"""Worked examples: Mayer path Betti numbers of network motifs and Laplacian spectra.

Writes examples/01_worked_examples/results/worked_examples.md.

Spectra are computed with orthonormal bases of Omega_n (inner product inherited
from the elementary paths), so (d^q)^* is the conjugate transpose.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from mayerpath import MayerPathComplex  # noqa: E402

MOTIFS = {
    "L1 (4-node feedback loop)": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "L2 (bi-parallel)": [(0, 1), (1, 2), (0, 3), (3, 2)],
    "L3 (bi-fan)": [(0, 2), (0, 3), (1, 2), (1, 3)],
}

DIGRAPHS = {
    "directed 3-cycle 0->1->2->0": (3, [(0, 1), (1, 2), (2, 0)]),
    "triangle 0->1->2, 0->2": (3, [(0, 1), (1, 2), (0, 2)]),
    "square 0->1->2, 0->3->2": (4, [(0, 1), (0, 3), (1, 2), (3, 2)]),
    "bi-fan 0->1, 0->3, 2->1, 2->3": (4, [(0, 1), (0, 3), (2, 1), (2, 3)]),
    "two squares on 6 vertices": (6, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]),
}


# Example 5.1 of the paper (label ex:lapvshom): same Mayer path Betti numbers,
# different Mayer path Laplacian spectra for N >= 3.
EXAMPLE_5_1 = {
    "G_A": [(0, 1), (0, 2), (0, 3), (1, 2), (3, 2)],
    "G_B": [(0, 2), (0, 3), (1, 0), (1, 2), (3, 2)],
}


def fmt(spec):
    return "{" + ", ".join(f"{x:.4g}" if abs(x) > 1e-10 else "0" for x in spec) + "}"


def example_5_1():
    """Lines of the Example 5.1 section; also checks the closed forms."""
    C = {(g, N): MayerPathComplex(range(4), E, N, 3)
         for g, E in EXAMPLE_5_1.items() for N in (2, 3, 4, 5, 6)}
    lines = ["## Example 5.1 (ex:lapvshom): same Betti numbers, different spectra", "",
             "G_A: 0->1, 0->2, 0->3, 1->2, 3->2;  G_B: 0->2, 0->3, 1->0, 1->2, 3->2", "",
             "| | dim Omega_0..3 (G_A) | dim Omega_0..3 (G_B) |", "|---|---|---|",
             f"| | {[C['G_A', 3].omega_dim(n) for n in range(4)]} "
             f"| {[C['G_B', 3].omega_dim(n) for n in range(4)]} |", "",
             "| N, q | beta_0, beta_1, beta_2 (G_A) | beta_0, beta_1, beta_2 (G_B) |", "|---|---|---|"]
    for N, q in [(2, 1), (3, 1), (3, 2)]:
        a = C["G_A", N].betti_numbers(q)[:3]
        b = C["G_B", N].betti_numbers(q)[:3]
        assert a == b
        lines.append(f"| N={N}, q={q} | {a} | {b} |")
    lines += ["", "| Laplacian | G_A | G_B |", "|---|---|---|"]
    for N in (2, 3):
        for q in range(1, N):
            for n in range(3):
                lines.append(f"| L^{N}_{{{n},{q}}} | {fmt(C['G_A', N].spectrum(n, q))} "
                             f"| {fmt(C['G_B', N].spectrum(n, q))} |")
    # closed forms: G_B, N = 3 and L^N_{2,2} = c^2{1,5} vs c^2{3-c, 3+c}, c = 2cos(pi/N)
    s2, s3 = np.sqrt(2), np.sqrt(3)
    assert np.allclose(C["G_B", 3].spectrum(0, 2), sorted([2 - s3, 3 - s2, 3 + s2, 2 + s3]))
    lines += ["", "For G_B and N = 3, Spec L_{0,2} = {2 - sqrt3, 3 - sqrt2, 3 + sqrt2, 2 + sqrt3}"
              " (= Spec L_{1,1} without its zero).", "",
              "| N | c = 2cos(pi/N) | Spec L^N_{2,2}(G_A) | c^2 {1, 5} | Spec L^N_{2,2}(G_B) | c^2 {3 - c, 3 + c} |",
              "|---|---|---|---|---|---|"]
    for N in (3, 4, 5, 6):
        c = 2 * np.cos(np.pi / N)
        a, b = C["G_A", N].spectrum(2, 2), C["G_B", N].spectrum(2, 2)
        fa, fb = c * c * np.array([1, 5]), c * c * np.array([3 - c, 3 + c])
        assert np.allclose(a, fa) and np.allclose(b, fb)
        lines.append(f"| {N} | {c:.4g} | {fmt(a)} | {fmt(fa)} | {fmt(b)} | {fmt(fb)} |")
    return lines + [""]


def main():
    lines = ["# Worked examples", "",
             "## Betti numbers of network motifs", "",
             "| motif | H_0^{2,1} | H_1^{2,1} | H_0^{3,1} | H_1^{3,1} | H_0^{3,2} | H_1^{3,2} |",
             "|---|---|---|---|---|---|---|"]
    for name, E in MOTIFS.items():
        b2 = MayerPathComplex(range(4), E, 2, 1).betti_numbers(1)
        C3 = MayerPathComplex(range(4), E, 3, 1)
        b31, b32 = C3.betti_numbers(1), C3.betti_numbers(2)
        lines.append(f"| {name} | {b2[0]} | {b2[1]} | {b31[0]} | {b31[1]} | {b32[0]} | {b32[1]} |")
    lines += ["", "## Spectra of the Mayer path Laplacians L_{n,q}", ""]
    for name, (nv, E) in DIGRAPHS.items():
        lines += [f"### {name}", "", "| N | n | q | dim Omega_n | spectrum of L_{n,q} |", "|---|---|---|---|---|"]
        for N in (2, 3, 5):
            C = MayerPathComplex(range(nv), E, N, 2)
            for n in range(3):
                if C.omega_dim(n) == 0:
                    continue
                for q in range(1, N):
                    try:
                        spec = fmt(C.spectrum(n, q))
                    except NotImplementedError:
                        spec = f"not computed (needs Omega_{n + N - q}; Omega_n is built for n <= 3)"
                    lines.append(f"| {N} | {n} | {q} | {C.omega_dim(n)} | {spec} |")
        lines.append("")
    lines += example_5_1()
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "worked_examples.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    print("wrote", os.path.join(out, "worked_examples.md"))


if __name__ == "__main__":
    main()
