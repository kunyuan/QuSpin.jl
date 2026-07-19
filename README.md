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

The implementation currently prioritizes semantic completeness and a small
native design. Some large sparse workloads still use dense reference kernels;
those backends can be replaced with specialized sparse/Krylov implementations
without changing the public Julia API.
