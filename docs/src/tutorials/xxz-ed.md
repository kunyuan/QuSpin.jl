# XXZ exact diagonalization

This tutorial constructs a periodic spin-1/2 XXZ Hamiltonian in a fixed
magnetization sector and computes its lowest eigenpairs. The workflow uses
native CSC storage from basis action through ARPACK.

## Model

We use

```math
H = \sum_i \left[\frac{J}{2}(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+)
    + \Delta S_i^zS_{i+1}^z\right].
```

```@example xxz
using QuSpin
using LinearAlgebra
using SparseArrays

L = 10
J = 1.0
Delta = 0.7
basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
bonds = [(1.0, i, mod1(i + 1, L)) for i in 1:L]

terms = [
    OperatorTerm("+-", [(J / 2, i, j) for (_, i, j) in bonds]),
    OperatorTerm("-+", [(J / 2, i, j) for (_, i, j) in bonds]),
    OperatorTerm("zz", [(Delta, i, j) for (_, i, j) in bonds]),
]

H = Hamiltonian(basis, terms; static_fmt=:csc)
@assert H.data isa SparseMatrixCSC
@assert H.hermitian === true
(dimension=length(basis), nonzeros=nnz(H.data))
```

## Lowest eigenpairs

```@example xxz
energies, vectors = eigsh(H; k=4, which=:SA, tol=1e-11)
residuals = [norm(H.data * vectors[:, i] - energies[i] * vectors[:, i])
             for i in eachindex(energies)]
@assert maximum(residuals) < 1e-8
(energies=energies, maximum_residual=maximum(residuals))
```

## Add a momentum sector

Translation symmetry is selected at basis construction. The reduced operator
is defined by the same physical projector ``P`` used for state lifting:

```@example xxz
momentum_basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false, kblock=0)
Hk = Hamiltonian(momentum_basis, terms; static_fmt=:csc)
P = projection_matrix(momentum_basis, ComplexF64; pcon=true)
parent = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
Hparent = Hamiltonian(parent, terms; static_fmt=:csc)

@assert Matrix(Hk) ≈ P' * Matrix(Hparent) * P atol=1e-11
(sector_dimension=length(momentum_basis), parent_dimension=length(parent))
```

For large calculations, avoid materializing `Matrix(H)` or the projector in
the timed path; they are used above only as small-system validation oracles.
