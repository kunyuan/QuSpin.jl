# Getting started

## Installation

QuSpin.jl currently installs directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/matrixlab-research/QuSpin.jl")
```

For development, clone the repository and instantiate its environment:

```sh
git clone https://github.com/matrixlab-research/QuSpin.jl.git
cd QuSpin.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Choose a basis

`SpinBasis1D` represents spin chains. A fixed `nup` sector reduces the Hilbert
space before any matrix is constructed:

```@example getting_started
using QuSpin

full = SpinBasis1D(8; pauli=false)
sector = SpinBasis1D(8; nup=4, pauli=false)
@assert length(full) == 2^8
@assert length(sector) == binomial(8, 4)
(full=length(full), fixed_magnetization=length(sector))
```

Other constructors cover bosons, spinless and spinful fermions, higher-spin
sites, tensor products, photon spaces, and callback-defined bases. Use a
`*BasisGeneral` type when a model needs user-supplied finite-order lattice
maps rather than the built-in one-dimensional translation and parity blocks.

## Define operator terms

An `OperatorTerm` combines an operator string with coupling rows. Sites are
one-based. Multi-site operators act on kets from right to left, which is
important for fermionic signs.

```@example getting_started
L = 8
basis = SpinBasis1D(L; nup=4, pauli=false)
nearest_neighbors = [(1.0, i, i + 1) for i in 1:(L - 1)]
terms = [
    OperatorTerm("+-", [(0.5, i, j) for (_, i, j) in nearest_neighbors]),
    OperatorTerm("-+", [(0.5, i, j) for (_, i, j) in nearest_neighbors]),
    OperatorTerm("zz", nearest_neighbors),
]
H = Hamiltonian(basis, terms; static_fmt=:csc)
size(H)
```

## Select storage honestly

Hamiltonians support `:dense`, `:csc`, `:csr`, and `:dia`. Sparse construction
is native; requesting CSC does not first construct a dense matrix.

```@example getting_started
using SparseArrays

Hcsc = as_sparse_format(H; static_fmt=:csc)
Hcsr = as_sparse_format(H; static_fmt=:csr)
Hdense = as_dense_format(H)

@assert Hcsc.data isa SparseMatrixCSC
@assert Hcsr.data isa SparseMatrixCSR
@assert Matrix(Hcsc) == Matrix(Hcsr) == Matrix(Hdense)
```

Use dense storage for complete eigensystems of small spaces. Prefer CSC or a
`QuantumLinearOperator` for selected eigenpairs and Krylov evolution.

## Validate numerical results

Always check solver residuals rather than relying only on returned status:

```@example getting_started
using LinearAlgebra

values, vectors = eigsh(Hcsc; k=3, which=:SA)
residual = norm(Matrix(Hcsc) * vectors - vectors * Diagonal(values))
@assert residual < 1e-9
residual
```
