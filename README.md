# Mayer Path Homology – computations

Code for computing **Mayer path homology**, **Mayer path Laplacians** and their
**persistent** versions for directed graphs, together with the scripts that
reproduce the examples of the paper

> D. Karaguler and G.-W. Wei, *Persistent Mayer Path Homology and Laplacians for Directed Networks* (2026).

Mayer path homology replaces the condition ∂² = 0 of path homology by ∂ᴺ = 0,
using an N-th root of unity in the boundary operator. N = 2 gives ordinary
(persistent) path homology and the path Laplacian.

## Installation

Python ≥ 3.9 with NumPy and SciPy; the examples also use Matplotlib.

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e ".[examples,test]"                  # or: pip install -r requirements.txt
```

The example scripts also run without installing the package: they add `src/`
to the Python path themselves.

## Quick start

```python
from mayerpath import MayerPathComplex, Filtration

# bi-parallel motif 0 -> 1 -> 2, 0 -> 3 -> 2
edges = [(0, 1), (1, 2), (0, 3), (3, 2)]

C = MayerPathComplex(vertices=range(4), edges=edges, N=3, max_degree=1)
C.betti_numbers(q=1)        # [3, 1]   beta_n^{3,1}, n = 0, 1
C.betti_numbers(q=2)        # [1, 3]
C.spectrum(n=0, q=2)        # eigenvalues of the Mayer path Laplacian L_{0,2}

# a filtration G_0 ⊂ G_1 ⊂ G_2
steps = [(range(4), []), (range(4), edges[:2]), (range(4), edges)]
F = Filtration(steps, N=3, max_degree=1)
F.betti_curve(n=0, q=1)     # beta_0^{3,1} at each step
F.betti(n=0, q=1, a=1, b=2) # persistent Betti number beta_0^{3,1}(1, 2)
F.spectrum(n=0, q=1, a=1, b=2)  # spectrum of the persistent Laplacian L^{1,2}_{0,1}
F.barcode(n=0, q=1)         # bars (birth step, death step)
```

## Conventions

For a digraph G with allowed elementary paths A_n, N ≥ 2 and ξ = e^{2πi/N}:

| object | definition |
|---|---|
| boundary (non-regular) | ∂ e_{i₀…iₙ} = Σⱼ ξʲ e_{i₀…îⱼ…iₙ} |
| ∂-invariant paths | Ωₙᴺ = { v ∈ Aₙ : ∂^q v ∈ A_{n−q}, q = 1,…,N−1 } |
| Mayer path homology | Hₙ^{N,q} = ker(∂^q on Ωₙ) / im(∂^{N−q} on Ω_{n+N−q}), 1 ≤ q ≤ N−1 |
| Mayer path Laplacian | L_{n,q} = (∂^q)\* ∂^q + ∂^{N−q} (∂^{N−q})\* on Ωₙ |
| persistent Laplacian | L^{t,s}_{n,q} = (∂_t^q)\* ∂_t^q + ∂^{N−q}_{t,s} (∂^{N−q}_{t,s})\*, with Ω_{n+N−q}(t,s) = { v ∈ Ω_{n+N−q}(G_s) : ∂^{N−q} v ∈ Ωₙ(G_t) } |
| persistent Betti number | βₙ^{N,q}(t,s) = rank(Hₙ^{N,q}(G_t) → Hₙ^{N,q}(G_s)) = dim ker L^{t,s}_{n,q} |

Ωₙ carries the inner product in which the elementary paths are orthonormal,
and all matrices are written in orthonormal bases of Ωₙ, so the adjoint is the
conjugate transpose. Ranks and kernels are computed numerically (SVD, relative
tolerance 1e-8).

**How Ωₙ is constructed (n ≤ 3).** Only Ω₀, …, Ω₃ are built; a quantity that
needs Ωₘ with m ≥ 4 raises `NotImplementedError` unless the digraph has no
allowed m-paths (then Ωₘ = 0). This covers the examples of the paper.

* Ω₀ and Ω₁ are spanned by the vertices and the edges.
* Ω₂ (the same for every N): group the 2-paths e_{amb} by their endpoints
  (a, b). If a → b, every e_{amb} is a triangle in Ω₂; otherwise Ω₂ contains
  exactly the combinations with zero coefficient sum (squares
  e_{amb} − e_{am'b}). An orthonormal basis is written down directly.
* Ω₃: the condition ∂v ∈ A₂ only involves interior deletions, which keep the
  endpoints, so Ω₃^{N,1} splits into one small cluster per endpoint pair
  (a, b). For N ≥ 3, the characterization of Ω₃ᴺ in the paper replaces the
  condition ∂²v ∈ A₁ by one linear condition per cluster with a ↛ b: the
  coefficients sum to zero. Clusters are solved by a small SVD each. If some
  3-path repeats a vertex (directed 2- or 3-cycles), where the
  characterization does not apply, ∂²v ∈ A₁ is imposed directly.

The tests compare these constructions with the definition of Ωₙᴺ on random
digraphs (acyclic and with cycles, N = 2, …, 5).

**Closed formulas for N = 3.** Following Section "Computing Mayer Path Homology
when N = 3" of the paper (`src/mayerpath/formulas.py`):

* square classes 𝒞 = V/∼, where i ∼ j when i, j are the middle vertices of a
  square e_{kil} − e_{kjl} (union–find over the 2-paths grouped by endpoints);
* the triangle matrix T: K^𝒞 → K^Δ, (TF)_{(i,j,k)} = F([k]) + ξF([j]) + ξ²F([i]);
* rank(∂₁∂₂ on Ω₂) = |V| − |𝒞| + rank T, hence
  β₀^{3,1} = |𝒞| − rank T and
  β₂^{3,2} = dim Ω₂ − |V| + |𝒞| − rank T − rank(∂₃ on Ω₃),
  with dim Ω₂ = |Δ| + Σ_{(k,l)∉E} (m_{kl} − 1).

`MayerPathComplex.betti` uses these formulas for β₀^{3,1} and β₂^{3,2}
(`method="rank"` forces the rank computation); the tests check them against
the ranks on random digraphs with self-loops and opposite edges. Digraphs may
contain self-loops (k, k) ∈ E, as in the paper.

**Where SVD is used.** (i) the null space of each small Ω₃ cluster,
(ii) numerical ranks of the boundary matrices (Betti numbers),
(iii) the null space defining Ω_{n+N−q}(t, s) for persistent Laplacians.
Laplacian spectra use a Hermitian eigensolver.

## Repository layout

```
src/mayerpath/
    core.py          allowed paths, Mayer boundary, explicit bases of Omega_0..Omega_3, ranks
    formulas.py      square classes, triangle matrix T, closed formulas for N = 3
    complex.py       MayerPathComplex: Betti numbers, Laplacians, spectra
    persistence.py   Filtration, persistent Betti numbers, persistent Laplacians, barcodes
    molecules.py     MOL2 reader and electronegativity digraphs of molecules
data/
    cb7.mol2         cucurbit[7]uril (see data/README.md)
examples/
    01_worked_examples/   network motifs and Laplacian spectra
    02_trapezohedron/     filtered trapezohedron: Betti tables, persistence, barcodes
    03_cb7/               application to cucurbit[7]uril
tests/
    test_mayerpath.py
```

## Reproducing the paper

| paper | command | output |
|---|---|---|
| Table: path vs Mayer path homology of L₁, L₂, L₃; Laplacian examples; Example 5.1 (G_A, G_B: same Betti numbers, different spectra) | `python examples/01_worked_examples/run_worked_examples.py` | `examples/01_worked_examples/results/worked_examples.md` |
| Example: trapezohedron with added edges (table, barcodes) | `python examples/02_trapezohedron/run_trapezohedron.py` | `examples/02_trapezohedron/results/trapezohedron.md`, `figures/trapezohedron_barcodes.pdf` |
| Application: cucurbit[7]uril (curves and numbers) | `python examples/03_cb7/run_cb7.py` then `python examples/03_cb7/plot_cb7.py` | `examples/03_cb7/results/*.csv`, `figures/cb7_betti.pdf`, `figures/cb7_spectra.pdf` |
| Application: the CB7 digraph filtration (atoms by element, directed edges) | `python examples/03_cb7/plot_cb7_filtration.py` | `figures/cb7_filtration.pdf` (whole molecule), `figures/cb7_filtration_unit.pdf` (one glycoluril unit) |
| Application in the paper: CB7 with the FRI distance d = 1 − exp(−(r/η)^κ), η = τ(r_i + r_j) | `python examples/03_cb7/run_cb7_fri.py`, then `python examples/03_cb7/plot_cb7.py --fri` and `python examples/03_cb7/plot_cb7_filtration.py --fri` | `results/cb7_fri_curves.csv`, `results/cb7_fri_events.md`, `results/cb7_fri_persistent_betti.csv`, `figures/cb7_fri_betti.pdf`, `figures/cb7_fri_spectra.pdf`, `figures/cb7_fri_filtration_unit.pdf` |

`run_cb7.py` takes about 1–2 minutes on a laptop; the other scripts take a few seconds.

### Cucurbit[7]uril digraphs

Following R. Wang and G.-W. Wei, *Persistent path Laplacian* (2023), atoms are
vertices; two atoms of different elements at distance ≤ 2r are joined by an
edge directed by Pauling electronegativity (H → C → N → O); isolated atoms are
removed. With this construction N = 2 reproduces their Betti numbers
(β₀ = 28, 42, 1 at r = 0.565, 0.710, 0.8 Å and the jump of β₂ at r = 1.68 Å),
which is checked in the tests. The digraphs are acyclic, so the non-regular
boundary used here agrees with their regular boundary.
`electronegativity_digraph(..., same_element="index")` instead joins atoms of
the same element as well, directed by atom index, and
`electronegativity_digraph(..., distance=fri_distance(elements, coords, tau, kappa))`
uses the FRI distance of Xia, Opron and Wei. The FRI distance is an increasing
function of r_ij / (r_i + r_j) (van der Waals radii), so τ and κ only relabel
the filtration values; `run_cb7_fri.py` therefore indexes it by that ratio.

## Tests

```bash
python -m pytest
```

The tests check ∂ᴺ = 0, the explicit Ω₂ and Ω₃ against the definition (including the two-γ₈ example), the motif table and the
Laplacian spectra worked out in the paper, ker L = H on random digraphs, the
trapezohedron example (including persistent Betti numbers and the kernel of the
persistent Laplacian), and the cucurbit[7]uril values of Wang and Wei.

## Citation

See `CITATION.cff`. If you use this code, please cite the paper above.

## License

MIT, see `LICENSE`.
