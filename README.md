# QuSpin.jl

[![CI](https://github.com/matrixlab-research/QuSpin.jl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/matrixlab-research/QuSpin.jl/actions/workflows/test.yml)
[![Documentation](https://github.com/matrixlab-research/QuSpin.jl/actions/workflows/docs.yml/badge.svg?branch=main)](https://matrixlab-research.github.io/QuSpin.jl/dev/)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia)](https://julialang.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Julia-native reconstruction of QuSpin's documented public API, generated and
verified through Minos. Python QuSpin is used only as an offline oracle; this
package has no Python runtime dependency.

The scientific behavior is cross-checked against the BSD-3-Clause
[QuSpin](https://github.com/QuSpin/QuSpin) project. This repository is an
independent Julia implementation rather than a Python binding.

## Installation

QuSpin.jl currently installs directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/matrixlab-research/QuSpin.jl")
```

The package supports Julia 1.10 and later. See the
[development documentation](https://matrixlab-research.github.io/QuSpin.jl/dev/) for the
getting-started guide, complete tutorials, cookbook, benchmarks, and generated
API reference.

## Compatibility coverage

The current migration covers the frozen QuSpin 1.0.1 denominator:

- 64 top-level objects: 20 classes, 40 functions, and 4 values;
- 282 public methods (excluding constructors) and 180 public attributes;
- spin, boson, spinless/spinful fermion, photon, tensor, and user-defined bases;
- arbitrary finite-order general-lattice maps, higher-spin local spaces, and
  deferred or threaded basis construction;
- static and time-dependent Hamiltonians, parameterized operators, and
  matrix-exponential actions;
- evolution, Lanczos, measurements, Floquet, and symmetry-block tools;
- dynamical correlators, Lehmann/Krylov response spectra, and gauge-invariant
  eigenspace tracking;
- prefix-pruned constrained state generation and matrix-free Lindblad
  density-matrix evolution;
- native and Python-compatible `QuantumOperator` persistence.

Python names are translated to Julia conventions where appropriate: types use
CamelCase, mutating operations end in `!`, and lattice sites are one-based.
The compatibility map and exact frozen denominator live in the Minos QuSpin
campaign; the independent verification repository checks the public package
without importing Python.

## Quick start

```julia
using QuSpin, LinearAlgebra

basis = SpinBasis1D(4; nup=2, pauli=false)
terms = [OperatorTerm("zz", [(1.0, 1, 2)])]
H = Hamiltonian(basis, terms)
eigvals(H)
```

For a sparse partial spectrum:

```julia
H = Hamiltonian(basis, terms; static_fmt=:csc)
values, vectors = eigsh(H; k=2, which=:SA)
@assert norm(Matrix(H) * vectors - vectors * Diagonal(values)) < 1e-9
```

## Dense and sparse storage

`Hamiltonian` and `QuantumOperator` support four honest storage paths:
Julia `Matrix` (`:dense`), `SparseArrays.SparseMatrixCSC` (`:csc`),
`SparseMatrixCSR` (`:csr`), and `DIAMatrix` (`:dia`). The latter two are
Julia-native adapters with real row-compressed and diagonal storage; neither
relabels CSC data. Dynamic terms can select their format independently, and
`eigsh` uses ARPACK's iterative sparse eigensolver.

```julia
using SparseArrays

H = Hamiltonian(basis, terms; static_fmt=:csc)
@assert H.data isa SparseMatrixCSC
lowest, states = eigsh(H; k=4, which=:SA)

Hcsr = as_sparse_format(H; static_fmt=:csr)
@assert Hcsr.data isa SparseMatrixCSR
```

## Physical symmetry sectors

One-dimensional spin, boson, spinless-fermion, and spinful-fermion bases
construct orthonormal physical sector projectors. Translation/momentum
(`a`, `kblock`) and parity (`pblock`) are shared; spin inversion blocks
(`zblock`, `pzblock`, `zAblock`, `zBblock`), boson particle-hole blocks
(`cblock`, `pcblock`, `cAblock`, `cBblock`), and spinful exchange blocks
(`sblock`, `psblock`) are also supported. Fermionic translation and parity
include the occupied-mode permutation sign.

```julia
k2 = SpinBasis1D(8; nup=4, pauli=false, kblock=2)
P = projection_matrix(k2, ComplexF64)
@assert P'P ≈ I
periodic_terms = [
    OperatorTerm("zz", [(1.0, site, mod1(site + 1, 8)) for site in 1:8]),
]
Hk2 = Hamiltonian(k2, periodic_terms)
```

Operators in a reduced basis are defined by the same projector,
`P' * O * P`. `check_symm`, `check_pcon`, and `check_hermitian` reject
incompatible operator lists; `representative`, `normalization`, and
`get_amp` use the actual orbit amplitudes.

`SpinBasisGeneral`, `BosonBasisGeneral`, `SpinlessFermionBasisGeneral`, and
`SpinfulFermionBasisGeneral` accept independent finite-order site maps, so
commuting two-dimensional translations and compatible point-group sectors do
not need to be flattened into one-dimensional built-ins. Spin bases also
support `S=1, 3/2, ...` with exact angular-momentum matrix elements.

## Operator archives

`save_zip(path, operator)` retains the versioned native Julia format.
`save_zip(path, operator; save_basis=false, format=:python)` writes the
dense/CSC NPZ layout used by Python QuSpin, and `load_zip` detects either
format. The archive dependencies are loaded only when this compatibility path
is used, so ordinary package startup and computations do not pay its load
cost.

## Matrix-free and Krylov paths

`QuantumLinearOperator` stores local terms and applies them directly to a
vector. It does not construct a dense or sparse matrix unless `Matrix(op)` is
explicitly requested. ARPACK partial eigensolves consume this matvec
directly.

Static `evolve`, `ExpmMultiplyParallel`, `BlockOps`, and Floquet step
propagation use an adaptive Arnoldi exponential action. A sparse Hamiltonian
therefore remains sparse through propagation; full dense diagonalization is
reserved for APIs that explicitly request a complete eigensystem, a dense
array, or the dense Floquet unitary itself.

Boson and fermion operator strings use the same right-to-left ket action and
Jordan-Wigner sign convention as the pinned Python oracle. Hamiltonians can be
constructed directly on spin, boson, spinless/spinful fermion, tensor,
photon, and user-defined bases.

The integration suite includes reduced, deterministic versions of workflows
used in the literature: random-field XXZ mid-spectrum states
(Pal–Huse, arXiv:1003.2613), sparse Lanczos quantum-quench evolution, and a
periodically driven spin chain (QuSpin paper, arXiv:1610.03042). These tests
validate the numerical workflow and storage path; they do not claim to
reproduce the papers' finite-size scaling results.

## Documentation

The documentation is built with
[Documenter.jl](https://documenter.juliadocs.org/) from source docstrings and
executable examples. Build it locally with:

```sh
julia --project=docs -e '
    using Pkg
    Pkg.develop(PackageSpec(path=pwd()))
    Pkg.instantiate()
'
julia --project=docs docs/make.jl
```

The strict build checks that every exported binding has a docstring and that
all tutorial examples execute successfully.

## Tests

Run the public unit, property, regression, and workflow suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Public tests cover all exported API families, storage representations,
symmetry and particle sectors, operator algebra, fermionic signs, Krylov
paths, time evolution, archive interchange, and paper-shaped workflows. A
separate verification repository retains held-out sizes, coefficients, and
Python-oracle observations.

## Benchmarks

The `benchmark/` environment contains reproducible `BenchmarkTools` workloads
for basis construction, native CSC Hamiltonian assembly, sparse matrix-vector
action, and iterative eigensolvers:

```sh
julia --project=benchmark -e '
    using Pkg
    Pkg.develop(path=pwd())
    Pkg.instantiate()
'
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    julia --project=benchmark benchmark/benchmarks.jl
```

Correctness checks run outside timed regions. Hosted-runner benchmarks are
observational and are not used as noisy pass/fail gates.

## Contributing

Changes to public behavior should include a focused unit or property test,
the relevant docstring update, and an independent verification case when they
close a semantic gap. Performance changes should report controlled storage,
thread counts, raw sample statistics, allocations, and physical residuals.

## License

QuSpin.jl is available under the [BSD 3-Clause License](LICENSE). The Python
QuSpin project is used only as an offline scientific-behavior oracle and is not
a runtime dependency.
