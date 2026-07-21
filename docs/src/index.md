# QuSpin.jl

QuSpin.jl is a Julia-native exact-diagonalization toolkit for finite quantum
many-body systems. It provides symmetry-aware bases, dense and sparse
Hamiltonians, matrix-free operators, iterative eigensolvers, time evolution,
Floquet tools, and measurements without a Python runtime dependency.

The package reconstructs the documented QuSpin 1.0.1 public API using Julia
conventions:

- lattice sites are one-based;
- mutating functions end in `!`;
- sparse matrices use native Julia CSC, CSR, or diagonal storage;
- Python QuSpin is used only as an offline verification oracle.

## Quick example

```@example quickstart
using QuSpin
using LinearAlgebra
using SparseArrays

L = 6
basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
bonds = [(1.0, site, mod1(site + 1, L)) for site in 1:L]
terms = [
    OperatorTerm("+-", [(0.5, i, j) for (_, i, j) in bonds]),
    OperatorTerm("-+", [(0.5, i, j) for (_, i, j) in bonds]),
    OperatorTerm("zz", bonds),
]
H = Hamiltonian(basis, terms; static_fmt=:csc)
energies, vectors = eigsh(H; k=2, which=:SA)

@assert H.data isa SparseMatrixCSC
@assert norm(Matrix(H) * vectors - vectors * Diagonal(energies)) < 1e-9
(dimension=length(basis), energies=energies)
```

## Where to go next

- [Getting started](@ref) explains installation, bases, operator strings, and
  storage choices.
- [XXZ exact diagonalization](@ref) builds and validates a complete sparse ED
  workflow.
- [Lanczos quantum quench](@ref) evolves a state without diagonalizing the full
  Hamiltonian.
- [Cookbook](@ref) gives short recipes for common basis and operator tasks.
- [Benchmarks](@ref) documents the reproducible performance protocol.
- [API reference](@ref) is generated from the package docstrings.

## Compatibility boundary

The frozen compatibility denominator contains 64 documented top-level objects,
282 methods excluding constructors, and 180 public attributes. Tests cover the
surface, numerical semantics, physical invariants, and representative
paper-shaped workflows. This does not make every undocumented Python behavior
part of the Julia API.
