# QuSpin.jl

Julia-native reconstruction of QuSpin's documented public API, generated and
verified through Minos. Python QuSpin is used only as an offline oracle; this
package has no Python runtime dependency.

The scientific behavior is cross-checked against the BSD-3-Clause
[QuSpin](https://github.com/QuSpin/QuSpin) project. This repository is an
independent Julia implementation rather than a Python binding.

The current migration covers the frozen QuSpin 1.0.1 denominator:

- 64 top-level objects: 20 classes, 40 functions, and 4 values;
- 282 public methods (excluding constructors) and 180 public attributes;
- spin, boson, spinless/spinful fermion, photon, tensor, and user-defined bases;
- static and time-dependent Hamiltonians, parameterized operators, and
  matrix-exponential actions;
- evolution, Lanczos, measurements, Floquet, and symmetry-block tools;
- Julia-native `QuantumOperator` persistence.

Python names are translated to Julia conventions where appropriate: types use
CamelCase, mutating operations end in `!`, and lattice sites are one-based.
The compatibility map and exact frozen denominator live in the Minos QuSpin
campaign; the independent verification repository checks the public package
without importing Python.

```julia
using QuSpin, LinearAlgebra

basis = SpinBasis1D(4; nup=2, pauli=false)
terms = [OperatorTerm("zz", [(1.0, 1, 2)])]
H = Hamiltonian(basis, terms)
eigvals(H)
```

## Dense and sparse storage

`Hamiltonian` and `QuantumOperator` support two real internal storage paths:
Julia `Matrix` (`:dense`) and `SparseArrays.SparseMatrixCSC` (`:csc`).
Construction with `static_fmt=:csc` assembles CSC triplets directly instead
of materializing a dense matrix first. Dynamic terms can independently use
`dynamic_fmt=:csc`, and `eigsh` uses ARPACK's iterative sparse eigensolver.

```julia
using SparseArrays

H = Hamiltonian(basis, terms; static_fmt=:csc)
@assert H.data isa SparseMatrixCSC
lowest, states = eigsh(H; k=4, which=:SA)
```

Julia's standard library does not provide CSR or DIA matrix types. Requests
for `:csr` or `:dia`, including `tocsr`, therefore fail explicitly rather than
returning a CSC matrix under the wrong name. Use `tocsc`, `as_sparse_format`,
or `update_matrix_formats!(H, :csc)`.

The integration suite includes reduced, deterministic versions of workflows
used in the literature: random-field XXZ mid-spectrum states
(Pal–Huse, arXiv:1003.2613), sparse Lanczos quantum-quench evolution, and a
periodically driven spin chain (QuSpin paper, arXiv:1610.03042). These tests
validate the numerical workflow and storage path; they do not claim to
reproduce the papers' finite-size scaling results.
