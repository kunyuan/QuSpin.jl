# QuSpin.jl

Julia-native reconstruction of selected QuSpin scientific workflows, generated
and verified through Minos. Python QuSpin is used only as an offline oracle;
this package has no Python runtime dependency.

The scientific behavior is cross-checked against the BSD-3-Clause
[QuSpin](https://github.com/QuSpin/QuSpin) project. This repository is an
independent Julia implementation rather than a Python binding.

The first slice implements:

- fixed-particle-number spin-one-half bases;
- one-based local operator terms;
- `z`, `zz`, `+-`, and `-+` Hamiltonians;
- dense matrix conversion and complete eigenspectra.

```julia
using QuSpin, LinearAlgebra

basis = SpinBasis1D(4; nup=2, pauli=false)
terms = [OperatorTerm("zz", [(1.0, 1, 2)])]
H = Hamiltonian(basis, terms)
eigvals(H)
```
