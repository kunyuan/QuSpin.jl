# Lanczos quantum quench

This tutorial evolves a Neel product state under the periodic Heisenberg
Hamiltonian using a Krylov subspace. Only Hamiltonian-vector actions are needed;
the complete eigensystem is never constructed.

```@example quench
using QuSpin
using LinearAlgebra

L = 10
basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
bonds = [(1.0, i, mod1(i + 1, L)) for i in 1:L]
terms = [
    OperatorTerm("+-", [(0.5, i, j) for (_, i, j) in bonds]),
    OperatorTerm("-+", [(0.5, i, j) for (_, i, j) in bonds]),
    OperatorTerm("zz", bonds),
]
H = Hamiltonian(basis, terms; static_fmt=:csc)

neel = sum(UInt64(1) << (site - 1) for site in 1:2:L)
psi0 = zeros(ComplexF64, length(basis))
psi0[state_index(basis, neel)] = 1
@assert norm(psi0) == 1
```

Build a Lanczos decomposition once and reuse it for several times:

```@example quench
E, V, Q_T = lanczos_full(H, psi0, 40; full_ortho=true)
times = 0.0:0.1:0.5
states_t = [expm_lanczos(E, V, Q_T; a=-im * t) for t in times]
@assert all(isapprox(norm(psi), 1; atol=1e-11) for psi in states_t)
```

The staggered magnetization is a useful quench observable:

```@example quench
imbalance = Hamiltonian(
    basis,
    [OperatorTerm("z", [(2(-1)^(site + 1) / L, site) for site in 1:L])];
    static_fmt=:csc,
)
values = [real(expt_value(imbalance, psi)) for psi in states_t]
@assert values[1] ≈ 1 atol=1e-14
collect(zip(times, values))
```

For a single static Hamiltonian, [`evolve`](@ref) is a higher-level alternative.
The explicit Lanczos path is useful when the same decomposition is reused for
many times or when convergence with the Krylov dimension must be inspected.
