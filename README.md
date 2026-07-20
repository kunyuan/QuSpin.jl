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
