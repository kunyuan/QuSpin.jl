# Cookbook

## Common bases

```julia
SpinBasis1D(12; nup=6, pauli=false)
SpinBasis1D(12; nup=6, pauli=false, kblock=0, pblock=1)
BosonBasis1D(8; Nb=4, sps=3)
SpinlessFermionBasis1D(12; Nf=6)
SpinfulFermionBasis1D(6; Nf=(3, 3))
HOBasis(12)
```

Choose a fixed conserved sector before building an operator. This saves both
construction time and memory, and makes forbidden particle-changing terms fail
early through `check_pcon`.

## Periodic nearest-neighbor couplings

```julia
bonds(L, value=1.0) = [(value, i, mod1(i + 1, L)) for i in 1:L]
```

For open boundaries, replace `1:L` with `1:(L - 1)` and use `i + 1`.

## Static versus time-dependent terms

```julia
drive(t, omega) = cos(omega * t)
static = Any[Any["zz", bonds(8)]]
dynamic = Any[Any["x", [(0.5, i) for i in 1:8], drive, (3.0,)]]
H = Hamiltonian(
    static,
    dynamic;
    basis=SpinBasis1D(8),
    dtype=ComplexF64,
    static_fmt=:csc,
    dynamic_fmt=:csc,
)
```

## Storage conversions

```julia
Hcsc = as_sparse_format(H; static_fmt=:csc)
Hcsr = as_sparse_format(H; static_fmt=:csr)
Hdia = as_sparse_format(H; static_fmt=:dia)
Hdense = as_dense_format(H)
```

Use CSC for Julia sparse linear algebra and ARPACK. CSR is useful for explicit
row-oriented kernels and interchange. DIA is effective only when the number of
occupied diagonals stays small.

## Selected versus complete spectra

```julia
lowest, vectors = eigsh(Hcsc; k=6, which=:SA)
all_values, all_vectors = eigh(Hdense)
```

`eigh` performs dense ``O(D^3)`` work and stores ``O(D^2)`` data. Use `eigsh`
or `aslinearoperator` when only a few states are required.

## Check a symmetry projection

```julia
reduced = SpinBasis1D(8; nup=4, pauli=false, kblock=0)
parent = SpinBasis1D(8; nup=4, pauli=false)
P = projection_matrix(reduced, ComplexF64)
@assert P' * P ≈ I
```

For production-scale runs, this explicit projector check belongs in a reduced
validation case rather than the main timed calculation.

## Entanglement entropy

```julia
result = ent_entropy(basis, psi; sub_sys_A=1:4, return_rdm="A")
entropy = result["Sent_A"]
rho_A = result["rdm_A"]
```

The state must use the same basis ordering as `basis`. Batched states are
accepted as columns where the relevant method documents matrix input.

## Reproducible iterative solves

Provide an explicit normalized initial vector and validate residuals:

```julia
v0 = normalize!(ones(Float64, length(basis)))
values, vectors = eigsh(H; k=4, which=:SA, v0=v0, tol=1e-10)
@assert norm(H * vectors - vectors * Diagonal(values)) < 1e-7
```
