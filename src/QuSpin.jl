module QuSpin

using LinearAlgebra
using SparseArrays

include("basis.jl")
include("operators.jl")
include("tools.jl")

using .Basis:
    AbstractBasis,
    BosonBasis1D,
    BosonBasisGeneral,
    FixedUInt,
    PhotonBasis,
    SpinBasis1D,
    SpinBasisGeneral,
    SpinfulFermionBasis1D,
    SpinfulFermionBasisGeneral,
    SpinlessFermionBasis1D,
    SpinlessFermionBasisGeneral,
    TensorBasis,
    UserBasis,
    UInt256,
    UInt1024,
    UInt4096,
    UInt16384,
    basis_int_to_python_int,
    basis_ones,
    basis_zeros,
    bitwise_and,
    bitwise_leftshift,
    bitwise_not,
    bitwise_or,
    bitwise_rightshift,
    bitwise_xor,
    coherent_state,
    ent_entropy,
    expanded_form,
    get_vec,
    get_basis_type,
    photon_Hspace_dim,
    projection_matrix,
    project_from,
    partial_trace,
    python_int_to_basis_int,
    int_to_state,
    check_hermitian,
    check_pcon,
    check_symm,
    get_amp,
    inplace_op!,
    make_basis!,
    make_basis_blocks,
    normalization,
    op_bra_ket,
    op_shift_sector,
    operator_matrix,
    representative,
    state_to_int,
    state_at,
    state_index,
    states
using .Operators:
    ExpOp,
    Hamiltonian,
    OperatorTerm,
    QuantumLinearOperator,
    QuantumOperator,
    anti_commutator,
    as_dense_format,
    aslinearoperator,
    as_sparse_format,
    apply,
    astype,
    commutator,
    check_is_dense,
    diagonal,
    eigh,
    eigsh,
    evolve,
    expt_value,
    get_mat,
    get_operators,
    isexp_op,
    ishamiltonian,
    isquantum_LinearOperator,
    isquantum_operator,
    load_zip,
    matrix_ele,
    project_to,
    quant_fluct,
    right_apply,
    rotate_by,
    sandwich,
    save_zip,
    set_a!,
    set_grid!,
    set_iterate!,
    set_diagonal!,
    unset_grid!,
    toarray,
    tohamiltonian,
    tocsc,
    tocsr,
    todense,
    update_matrix_formats!
using .Tools:
    ExpmMultiplyParallel,
    BlockOps,
    Floquet,
    FloquetTimeVector,
    ftlm_static_iteration,
    ltlm_static_iteration,
    ed_state_vs_time,
    diag_ensemble,
    expm_lanczos,
    array_to_ints,
    block_diag_hamiltonian,
    block_expm,
    compute_all_blocks!,
    get_matvec_function,
    get_coordinates,
    ints_to_array,
    kl_div,
    lanczos_full,
    lanczos_iter,
    lin_comb_Q_T,
    matvec,
    mean_level_spacing,
    obs_vs_time,
    project_op,
    update_blocks!,
    expm_multiply_parallel

export Basis, Operators, Tools
export AbstractBasis, ExpOp, FixedUInt, Hamiltonian, OperatorTerm
export BosonBasis1D, BosonBasisGeneral
export PhotonBasis, TensorBasis
export UserBasis
export SpinfulFermionBasis1D, SpinfulFermionBasisGeneral
export SpinlessFermionBasis1D, SpinlessFermionBasisGeneral
export ExpmMultiplyParallel, expm_multiply_parallel
export BlockOps, block_diag_hamiltonian, block_expm
export compute_all_blocks!, update_blocks!
export Floquet, FloquetTimeVector, get_coordinates
export QuantumLinearOperator, QuantumOperator, SpinBasis1D
export SpinBasisGeneral
export load_zip, save_zip
export UInt256, UInt1024, UInt4096, UInt16384
export basis_int_to_python_int, basis_ones, basis_zeros
export bitwise_and, bitwise_leftshift, bitwise_not, bitwise_or
export bitwise_rightshift, bitwise_xor
export coherent_state, get_basis_type, photon_Hspace_dim
export ent_entropy, partial_trace, projection_matrix, python_int_to_basis_int
export expanded_form, get_vec, project_from
export check_hermitian, check_pcon, check_symm, get_amp, inplace_op!
export make_basis!, make_basis_blocks, normalization
export op_bra_ket, op_shift_sector, operator_matrix, representative
export anti_commutator, commutator
export apply, get_mat, isexp_op, right_apply, sandwich
export set_a!, set_grid!, set_iterate!, unset_grid!
export as_dense_format, as_sparse_format, astype, diagonal, eigh, eigsh
export aslinearoperator, check_is_dense
export evolve, expt_value, ishamiltonian, matrix_ele, project_to, quant_fluct
export isquantum_LinearOperator, set_diagonal!
export get_operators, isquantum_operator, tohamiltonian
export rotate_by, toarray, tocsc, tocsr, todense, update_matrix_formats!
export array_to_ints, get_matvec_function, ints_to_array, kl_div, matvec
export mean_level_spacing, project_op
export diag_ensemble, obs_vs_time
export ed_state_vs_time, expm_lanczos, lanczos_full, lanczos_iter, lin_comb_Q_T
export ftlm_static_iteration, ltlm_static_iteration
export eigvals, state_at, state_index, states
export int_to_state, state_to_int

end
