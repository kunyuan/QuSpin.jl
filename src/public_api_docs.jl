# Public docstrings for compatibility helpers and overloaded API families.
# Core types keep their implementation-level docstrings next to their
# definitions; this file documents the cross-cutting public surface in one
# discoverable place for `?name` and Documenter.jl.

@doc raw"""
    isbasis(value) -> Bool

Return `true` when `value` is a QuSpin [`AbstractBasis`](@ref).
""" isbasis

@doc raw"""
    states(basis)

Return the encoded states represented by `basis` in its canonical basis order.
The returned collection is safe for callers to inspect without mutating the
basis.
""" states

@doc raw"""
    state_at(basis, index)

Return the encoded state at the one-based basis `index`.
""" state_at

@doc raw"""
    state_index(basis, state)

Return the one-based index of an encoded `state` in `basis`. An `ArgumentError`
is thrown when the state is outside the selected sector.
""" state_index

@doc raw"""
    int_to_state(basis, state; kwargs...) -> String

Format an encoded integer state using the local-state convention of `basis`.
""" int_to_state

@doc raw"""
    state_to_int(basis, state::AbstractString)

Parse a formatted basis state and return its encoded integer representation.
""" state_to_int

@doc raw"""
    projection_matrix(basis, [T=ComplexF64]; pcon=false, sparse=true)

Construct the isometry whose columns lift vectors from `basis` into the parent
computational space. Set `pcon=true` to target the parent particle sector when
supported.
""" projection_matrix

@doc raw"""
    project_from(basis, value; pcon=false, sparse=true)

Lift a vector or column-state matrix from reduced-basis coordinates into the
parent computational or particle-conserving space.
""" project_from

@doc raw"""
    project_to(basis, value; pcon=false)
    project_to(operator, basis; kwargs...)

Project states or operators into `basis`. The precise output follows the input
kind and preserves sparse storage where requested.
""" project_to

@doc raw"""
    get_vec(basis, value; kwargs...)

Compatibility alias for lifting a reduced-basis vector or matrix with
[`project_from`](@ref).
""" get_vec

@doc raw"""
    operator_matrix(basis, opstring, couplings; storage=:csc)

Assemble the operator described by `opstring` and its coupling rows in the
chosen basis. Operator sites are one-based and act on kets from right to left.
""" operator_matrix

@doc raw"""
    inplace_op!(output, basis, opstring, couplings)

Apply an operator string to basis-state data and write the transformed entries
to `output`.
""" inplace_op!

@doc raw"""
    op_bra_ket(basis, opstring, couplings)

Return the nonzero bra-state, ket-state, and matrix-element mapping generated
by a local operator string.
""" op_bra_ket

@doc raw"""
    expanded_form(basis, static=Any[], dynamic=Any[])

Expand composite local operators such as `x` and `y` into primitive raising
and lowering operator strings.
""" expanded_form

@doc raw"""
    check_hermitian(basis, static, dynamic=Any[]) -> Bool

Validate that an operator specification is Hermitian in the selected basis.
An `ArgumentError` describes the first incompatible term.
""" check_hermitian

@doc raw"""
    check_pcon(basis, static, dynamic=Any[]) -> Bool

Validate that an operator specification preserves the selected particle
sector.
""" check_pcon

@doc raw"""
    check_symm(basis, static, dynamic=Any[]) -> Bool

Validate that an operator specification preserves every symmetry block carried
by `basis`.
""" check_symm

@doc raw"""
    representative(basis, state)

Return the canonical orbit representative associated with an encoded state.
""" representative

@doc raw"""
    normalization(basis, state)

Return the symmetry-orbit normalization of `state` in `basis`.
""" normalization

@doc raw"""
    get_amp(basis, state)

Return the complex amplitude and representative information used to project a
computational state into a symmetry sector.
""" get_amp

@doc raw"""
    op_shift_sector(target_basis, source_basis, opstring, couplings, state)

Apply an operator between compatible source and target symmetry or particle
sectors.
""" op_shift_sector

@doc raw"""
    make_basis!(basis)

Materialize a basis created with deferred construction. The same object is
returned after its state and block tables are populated.
""" make_basis!

@doc raw"""
    make_basis_blocks(basis; kwargs...)

Construct and return the symmetry-block metadata associated with `basis`.
""" make_basis_blocks

@doc raw"""
    basis_int_to_python_int(value)

Convert a Julia basis integer, including [`FixedUInt`](@ref) values, to an
arbitrary-precision integer compatible with Python's integer range.
""" basis_int_to_python_int

@doc raw"""
    python_int_to_basis_int(value; dtype=nothing)

Convert a nonnegative arbitrary-precision integer to the smallest supported
basis-integer representation, or to the explicitly requested `dtype`.
""" python_int_to_basis_int

@doc raw"""
    basis_zeros(shape, dtype=UInt32)
    basis_ones(shape, dtype=UInt32)

Create arrays of basis integers filled with zero or one.
""" basis_zeros

@doc (@doc basis_zeros) basis_ones

const _bitwise_doc = raw"""
    bitwise_*(x...; out=nothing, where=nothing)

Apply an elementwise bitwise operation to ordinary or fixed-width basis
integers. `out` enables destination reuse and `where` selects written entries.
"""
@doc _bitwise_doc bitwise_and
@doc _bitwise_doc bitwise_or
@doc _bitwise_doc bitwise_xor
@doc _bitwise_doc bitwise_not
@doc _bitwise_doc bitwise_leftshift
@doc _bitwise_doc bitwise_rightshift

@doc raw"""
    BosonBasisGeneral(L; Nb=nothing, sps=nothing, symmetries=..., kwargs...)

Construct a bosonic basis with arbitrary compatible finite-order site maps.
Use `Nb` for a fixed particle sector and `sps` for the local cutoff.
""" BosonBasisGeneral

@doc raw"""
    SpinlessFermionBasisGeneral(L; Nf=nothing, symmetries=..., kwargs...)

Construct a spinless-fermion basis with user-supplied finite-order lattice
symmetries and the correct fermionic permutation signs.
""" SpinlessFermionBasisGeneral

@doc raw"""
    SpinfulFermionBasisGeneral(L; Nf=nothing, symmetries=..., kwargs...)

Construct a spinful-fermion basis with independent up/down occupations and
general compatible lattice symmetries.
""" SpinfulFermionBasisGeneral

@doc raw"""
    TensorBasis(factors...)

Form the tensor-product basis of two or more component bases. Operator strings
use `|` to separate factor-local actions.
""" TensorBasis

@doc raw"""
    PhotonBasis(particle_basis, N; Nph=nothing, Ntot=nothing)

Combine a particle basis with a truncated photon Fock space. `Nph` fixes the
photon cutoff; `Ntot` selects a conserved total-excitation sector.
""" PhotonBasis

@doc raw"""
    as_dense_format(operator; static_fmt=:dense, dynamic_fmt=:dense)
    as_sparse_format(operator; static_fmt=:csc, dynamic_fmt=:csc)

Return a copy of an operator converted to the requested dense or sparse storage
formats.
""" as_dense_format

@doc (@doc as_dense_format) as_sparse_format

const _operator_conversion_doc = raw"""
    toarray(operator; kwargs...)
    todense(operator; kwargs...)
    tocsc(operator; kwargs...)
    tocsr(operator; kwargs...)

Materialize an operator in the named dense, CSC, or CSR representation.
Parameterized operators accept `pars` through `kwargs`.
"""
@doc _operator_conversion_doc toarray
@doc _operator_conversion_doc todense
@doc _operator_conversion_doc tocsc
@doc _operator_conversion_doc tocsr

@doc raw"""
    astype(operator, T)

Return an operator whose stored coefficients and matrices use scalar type `T`.
""" astype

@doc raw"""
    diagonal(operator; kwargs...)

Return the operator diagonal without requiring callers to materialize a dense
matrix.
""" diagonal

@doc raw"""
    eigh(operator; kwargs...) -> (values, vectors)

Compute the complete Hermitian eigensystem. This is a dense operation intended
for modest Hilbert spaces.
""" eigh

@doc raw"""
    eigsh(operator; k=6, which=:SA, sigma=nothing, return_eigenvectors=true, kwargs...)

Compute selected eigenpairs with an iterative ARPACK solve. `which` accepts
`:SA`, `:LA`, `:SM`, `:LM`, or `:BE`; `sigma` enables shift-invert targeting.
""" eigsh

@doc raw"""
    aslinearoperator(operator; kwargs...)

Return a [`QuantumLinearOperator`](@ref) or compatible matrix-free view of an
operator for iterative algorithms.
""" aslinearoperator

@doc raw"""
    check_is_dense(operator) -> Bool

Return whether every stored component of an operator uses dense storage.
""" check_is_dense

@doc raw"""
    get_mat(operator; time=0, pars=nothing)

Return the concrete matrix represented by an operator at the requested time or
parameter point.
""" get_mat

@doc raw"""
    get_operators(operator, key=nothing)

Return all named components of a [`QuantumOperator`](@ref), or the component
selected by `key`.
""" get_operators

@doc raw"""
    tohamiltonian(operator; pars=Dict())

Evaluate a parameterized operator and return the corresponding
[`Hamiltonian`](@ref).
""" tohamiltonian

const _operator_predicate_doc = raw"""
Return `true` when the supplied value is an instance of the corresponding
QuSpin operator abstraction.
"""
@doc _operator_predicate_doc isexp_op
@doc _operator_predicate_doc ishamiltonian
@doc _operator_predicate_doc isquantum_LinearOperator
@doc _operator_predicate_doc isquantum_operator

@doc raw"""
    evolve(operator, state, t0, times; eom=:SE, iterate=false, kwargs...)

Evolve a state or density matrix over `times`. `eom=:SE` solves the
Schrödinger equation and `eom=:LvNE` evolves a density matrix.
""" evolve

@doc raw"""
    expt_value(operator, state; kwargs...)

Compute the expectation value of `operator` for a state vector or batch of
states.
""" expt_value

@doc raw"""
    matrix_ele(operator, left, right; kwargs...)

Compute the matrix element `left' * operator * right`.
""" matrix_ele

@doc raw"""
    quant_fluct(operator, state; kwargs...)

Compute the quantum variance ``\langle O^2\rangle-\langle O\rangle^2``.
""" quant_fluct

@doc raw"""
    right_apply(operator, state; kwargs...)

Apply an operator from the right to a row-state representation.
""" right_apply

@doc raw"""
    sandwich(operator, left, right=left; kwargs...)

Evaluate an operator sandwiched between left and right vectors or matrices.
""" sandwich

@doc raw"""
    rotate_by(operator, generator; a=1, generator_is_diagonal=false)

Return the similarity-transformed operator generated by an exponential action.
""" rotate_by

@doc raw"""
    set_diagonal!(operator, diagonal)

Replace the cached diagonal used by a [`QuantumLinearOperator`](@ref).
""" set_diagonal!

@doc raw"""
    update_matrix_formats!(operator, formats)

Convert selected static, dynamic, or parameterized components in place to the
requested storage formats.
""" update_matrix_formats!

@doc raw"""
    save_zip(path, operator; save_basis=true, format=:native)

Persist a [`QuantumOperator`](@ref) in QuSpin.jl's native archive or in the
Python-compatible dense/CSC NPZ layout.
""" save_zip

@doc raw"""
    set_grid!(expop, start, stop, num; endpoint=true)
    unset_grid!(expop)
    set_iterate!(expop, iterate=true)

Configure or clear the time grid and iterator mode of an [`ExpOp`](@ref).
""" set_grid!

@doc (@doc set_grid!) unset_grid!
@doc (@doc set_grid!) set_iterate!

@doc raw"""
    BlockOps(blocks; kwargs...)

Store a collection of symmetry blocks and the operators needed to assemble or
evolve them independently.
""" BlockOps

@doc raw"""
    compute_all_blocks!(blocks; kwargs...)

Construct every deferred block carried by [`BlockOps`](@ref).
""" compute_all_blocks!

@doc raw"""
    update_blocks!(blocks, time; kwargs...)

Update the time-dependent matrices cached by [`BlockOps`](@ref).
""" update_blocks!

@doc raw"""
    block_expm(blocks, state, times; kwargs...)

Evolve independent symmetry blocks with matrix-exponential actions and combine
the results in the requested representation.
""" block_expm

@doc raw"""
    expm_multiply_parallel(A, a=1; kwargs...)

Construct an [`ExpmMultiplyParallel`](@ref) action object for repeated
applications of ``\exp(aA)``.
""" expm_multiply_parallel

@doc raw"""
    get_matvec_function(A)

Return the storage-specialized matrix-vector kernel used for `A`.
""" get_matvec_function
