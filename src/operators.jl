module Operators

using Arpack
using LinearAlgebra
using Serialization
using SparseArrays
import ..Basis
using ..Basis:
    AbstractBasis,
    SpinBasis1D,
    check_hermitian,
    check_pcon,
    check_symm,
    operator_matrix

include("sparse_formats.jl")

export DIAMatrix, SparseMatrixCSR
export ExpOp, Hamiltonian, OperatorTerm, anti_commutator, commutator
export QuantumLinearOperator, isquantum_LinearOperator, set_diagonal!
export QuantumOperator, get_operators, isquantum_operator, tohamiltonian
export load_zip, save_zip
export apply, get_mat, isexp_op, right_apply, sandwich
export set_a!, set_grid!, set_iterate!, unset_grid!
export as_dense_format, as_sparse_format, astype, diagonal, eigh, eigsh
export aslinearoperator, check_is_dense
export evolve, expt_value, ishamiltonian, matrix_ele, project_to, quant_fluct
export rotate_by, toarray, tocsc, tocsr, todense, update_matrix_formats!

"""
    OperatorTerm(op, couplings)

A local spin operator. Each coupling is a tuple whose first element is the
coefficient and whose remaining elements are one-based lattice sites.
"""
struct OperatorTerm{C<:AbstractVector}
    op::String
    couplings::C
end

function OperatorTerm(op::AbstractString, couplings::AbstractVector)
    isempty(op) && throw(ArgumentError("operator string cannot be empty"))
    normalized = map(c -> Tuple(c), couplings)
    arity = count(character -> character != '|', op)
    for coupling in normalized
        length(coupling) == arity + 1 ||
            throw(ArgumentError("operator arity and coupling sites differ"))
        all(site -> site isa Integer, coupling[2:end]) ||
            throw(ArgumentError("all sites must be integers"))
    end
    return OperatorTerm(String(op), normalized)
end

"""
    Hamiltonian(basis, terms)

Many-body operator assembled in the enumeration of `basis`.
"""
const NativeMatrix{T} = Union{
    Matrix{T},
    SparseMatrixCSC{T,Int},
    SparseMatrixCSR{T,Int},
    DIAMatrix{T,Int},
}

mutable struct Hamiltonian{T<:Number}
    basis::AbstractBasis
    terms::Vector{OperatorTerm}
    data::NativeMatrix{T}
    dynamic_terms::Vector{Tuple{NativeMatrix{T},Any,Tuple}}
    hermitian::Union{Nothing,Bool}
end

function _consolidate_terms(terms::AbstractVector{<:OperatorTerm})
    coefficients = Dict{Tuple{String,Tuple},Any}()
    order = Tuple{String,Tuple}[]
    for term in terms, coupling in term.couplings
        key = (term.op, Tuple(coupling[2:end]))
        haskey(coefficients, key) || push!(order, key)
        coefficients[key] =
            get(coefficients, key, zero(first(coupling))) + first(coupling)
    end
    grouped = Dict{String,Vector{Tuple}}()
    op_order = String[]
    for key in order
        coefficient = coefficients[key]
        iszero(coefficient) && continue
        op, sites = key
        haskey(grouped, op) || begin
            grouped[op] = Tuple[]
            push!(op_order, op)
        end
        push!(grouped[op], (coefficient, sites...))
    end
    consolidated = OperatorTerm[]
    for op in op_order
        entries = grouped[op]
        coefficient_type =
            promote_type((typeof(first(entry)) for entry in entries)...)
        site_type = promote_type(
            (
                typeof(site)
                for entry in entries
                for site in Base.tail(entry)
            )...,
        )
        arity = length(first(entries)) - 1
        coupling_type = Core.apply_type(
            Tuple,
            coefficient_type,
            ntuple(_ -> site_type, arity)...,
        )
        couplings = Vector{coupling_type}(undef, length(entries))
        for (index, entry) in pairs(entries)
            couplings[index] = (
                convert(coefficient_type, first(entry)),
                (convert(site_type, site) for site in Base.tail(entry))...,
            )
        end
        push!(consolidated, OperatorTerm(op, couplings))
    end
    return consolidated
end

function _normalize_matrix_format(format; default=:dense)
    format === nothing && return default
    normalized = Symbol(lowercase(String(format)))
    normalized === :sparse && return :csc
    normalized in (:dense, :csc, :csr, :dia) && return normalized
    throw(ArgumentError("matrix format must be :dense, :csc, :csr, or :dia"))
end

function _matrix_with_format(
    data::AbstractMatrix,
    ::Type{T},
    format,
;
    copy_data::Bool=true,
) where {T<:Number}
    normalized = _normalize_matrix_format(format)
    converted = if T <: Real && eltype(data) <: Complex
        maximum(abs, imag.(data); init=0.0) <= 2e-11 ||
            throw(InexactError(:_matrix_with_format, T, data))
        real.(data)
    else
        data
    end
    if normalized === :dense
        return !copy_data && converted isa Matrix{T} ?
            converted :
            Matrix{T}(converted)
    elseif normalized === :csr
        return !copy_data && converted isa SparseMatrixCSR{T,Int} ?
            converted :
            SparseMatrixCSR(SparseMatrixCSC{T,Int}(sparse(converted)))
    elseif normalized === :dia
        return !copy_data && converted isa DIAMatrix{T,Int} ?
            converted :
            DIAMatrix(SparseMatrixCSC{T,Int}(sparse(converted)))
    end
    if !copy_data && converted isa SparseMatrixCSC{T,Int}
        return converted
    end
    return SparseMatrixCSC{T,Int}(sparse(converted))
end

_storage_format(data::Matrix) = :dense
_storage_format(data::SparseMatrixCSC) = :csc
_storage_format(data::SparseMatrixCSR) = :csr
_storage_format(data::DIAMatrix) = :dia

function _hamiltonian_from_data(
    basis::AbstractBasis,
    data::AbstractMatrix;
    format=_storage_format(data),
    copy::Bool=true,
)
    T = eltype(data)
    matrix = _matrix_with_format(data, T, format; copy_data=copy)
    return Hamiltonian{eltype(matrix)}(
        basis,
        OperatorTerm[],
        matrix,
        Tuple{NativeMatrix{eltype(matrix)},Any,Tuple}[],
        ishermitian(matrix),
    )
end

function _coefficient_type(terms)
    coefficient_types = Type[]
    complex_operator = false
    for term in terms
        complex_operator |= 'y' in term.op
        for coupling in term.couplings
            push!(coefficient_types, typeof(first(coupling)))
        end
    end
    isempty(coefficient_types) && return Float64
    scalar_type = promote_type(Float64, coefficient_types...)
    return complex_operator ? promote_type(ComplexF64, scalar_type) : scalar_type
end

@inline function _site_mask(basis::SpinBasis1D, site::Integer)
    1 <= site <= basis.L ||
        throw(ArgumentError("site $site lies outside 1:$(basis.L)"))
    return UInt64(1) << (Int(site) - 1)
end

function _apply_local(
    basis::SpinBasis1D,
    state::UInt64,
    op::Char,
    site::Integer,
)
    mask = _site_mask(basis, site)
    occupied = !iszero(state & mask)
    if op == 'I'
        return state, 1.0, true
    elseif op == 'z'
        scale = basis.pauli ? 1.0 : 0.5
        return state, occupied ? scale : -scale, true
    elseif op == '+'
        occupied && return state, 0.0, false
        scale = basis.pauli ? 2.0 : 1.0
        return state | mask, scale, true
    elseif op == '-'
        occupied || return state, 0.0, false
        scale = basis.pauli ? 2.0 : 1.0
        return state & ~mask, scale, true
    elseif op == 'x'
        scale = basis.pauli ? 1.0 : 0.5
        return xor(state, mask), scale, true
    elseif op == 'y'
        scale = basis.pauli ? 1.0 : 0.5
        return xor(state, mask), occupied ? -im * scale : im * scale, true
    else
        throw(ArgumentError("unsupported spin operator '$op'"))
    end
end

function _assemble_spin_term!(
    rows::Vector{Int},
    columns::Vector{Int},
    values::Vector{T},
    basis::SpinBasis1D,
    term::OperatorTerm{C},
) where {T,C}
    operator_length = length(term.op)
    for coupling in term.couplings
        coefficient = convert(T, first(coupling))
        sites = Base.tail(coupling)
        for (column, initial_state) in pairs(basis.encoded_states)
            amplitude = coefficient
            state = initial_state
            alive = true
            for operator_index in operator_length:-1:1
                op = term.op[operator_index]
                state, factor, alive = _apply_local(
                    basis,
                    state,
                    op,
                    sites[operator_index],
                )
                alive || break
                amplitude *= convert(T, factor)
            end
            alive || continue
            row = get(basis.lookup, state, 0)
            iszero(row) && continue
            push!(rows, row)
            push!(columns, column)
            push!(values, amplitude)
        end
    end
    return nothing
end

function _assemble(
    basis::AbstractBasis,
    terms::AbstractVector{<:OperatorTerm},
    ::Type{T},
    format=:dense,
) where {T}
    if !(basis isa SpinBasis1D) || Basis._has_symmetry(basis.symmetry)
        matrix = spzeros(ComplexF64, length(basis), length(basis))
        for term in terms
            matrix += sparse(
                operator_matrix(
                    basis,
                    term.op,
                    term.couplings;
                    sparse=true,
                ),
            )
        end
        return _matrix_with_format(matrix, T, format)
    end
    rows = Int[]
    columns = Int[]
    values = T[]
    for term in terms
        _assemble_spin_term!(rows, columns, values, basis, term)
    end
    normalized = _normalize_matrix_format(format)
    matrix = sparse(rows, columns, values, length(basis), length(basis))
    return _matrix_with_format(matrix, T, normalized)
end

function Hamiltonian(
    basis::AbstractBasis,
    terms::AbstractVector{<:OperatorTerm};
    static_fmt=:dense,
    check_symm::Bool=true,
    check_herm::Bool=true,
    check_pcon::Bool=true,
)
    normalized = OperatorTerm[terms...]
    hermitian = if check_herm
        structural =
            basis isa Union{SpinBasis1D,Basis.DiscreteBasis{:boson}} ?
            _structured_is_hermitian(normalized) :
            nothing
        structural === nothing ?
            _matrixfree_is_hermitian(basis, normalized) :
            structural
    else
        nothing
    end
    check_herm && !hermitian &&
        throw(ArgumentError("operator list is not Hermitian"))
    check_pcon && !Basis.check_pcon(basis, normalized, Any[]) &&
        throw(ArgumentError("operator list violates the selected particle sector"))
    check_symm && !Basis.check_symm(basis, normalized, Any[]) &&
        throw(ArgumentError("operator list violates the selected symmetry sector"))
    T = _coefficient_type(normalized)
    Basis._basis_requires_complex(basis) &&
        (T = promote_type(T, ComplexF64))
    return Hamiltonian{T}(
        basis,
        normalized,
        _assemble(basis, normalized, T, static_fmt),
        Tuple{NativeMatrix{T},Any,Tuple}[],
        hermitian,
    )
end

function _normalize_operator_terms(entries)
    terms = OperatorTerm[]
    matrices = AbstractMatrix[]
    for entry in entries
        if entry isa OperatorTerm
            push!(terms, entry)
        elseif entry isa AbstractMatrix
            push!(matrices, entry)
        elseif entry isa Tuple || entry isa AbstractVector
            length(entry) >= 2 ||
                throw(ArgumentError("operator entries require an operator and couplings"))
            push!(terms, OperatorTerm(String(entry[1]), collect(entry[2])))
        else
            throw(ArgumentError("unsupported Hamiltonian entry"))
        end
    end
    return terms, matrices
end

function _basis_from_constructor(N, basis, basis_kwargs)
    basis !== nothing && return basis
    N === nothing && throw(ArgumentError("N or basis is required"))
    allowed = Dict{Symbol,Any}()
    for (key, value) in basis_kwargs
        name = Symbol(key)
        name === :Nup && (name = :nup)
        name in (:nup, :pauli) && (allowed[name] = value)
    end
    return SpinBasis1D(N; allowed...)
end

"""
    Hamiltonian(static_list, dynamic_list; N=nothing, basis=nothing,
                dtype=ComplexF64, ...)

QuSpin-style constructor accepting native `OperatorTerm`s, `[op, couplings]`
entries, or matrices. Dynamic entries are `[op, couplings, f, f_args]` or
`[matrix, f, f_args]`.
"""
function Hamiltonian(
    static_list,
    dynamic_list;
    N=nothing,
    basis=nothing,
    shape=nothing,
    dtype::Type=ComplexF64,
    static_fmt=nothing,
    dynamic_fmt=nothing,
    copy::Bool=true,
    check_symm::Bool=true,
    check_herm::Bool=true,
    check_pcon::Bool=true,
    basis_kwargs...,
)
    selected_basis = _basis_from_constructor(N, basis, basis_kwargs)
    terms, matrices = _normalize_operator_terms(static_list)
    terms = _consolidate_terms(terms)
    static_format = _normalize_matrix_format(static_fmt)
    dynamic_default = dynamic_fmt isa AbstractDict || dynamic_fmt === nothing ?
        static_format :
        _normalize_matrix_format(dynamic_fmt; default=static_format)
    static_matrix = _assemble(selected_basis, terms, dtype, static_format)
    for matrix in matrices
        size(matrix) == size(static_matrix) ||
            throw(DimensionMismatch("static matrix has the wrong shape"))
        static_matrix += _matrix_with_format(matrix, dtype, static_format; copy_data=copy)
    end

    dynamic_terms = Tuple{NativeMatrix{dtype},Any,Tuple}[]
    for entry in dynamic_list
        (entry isa Tuple || entry isa AbstractVector) ||
            throw(ArgumentError("dynamic entries must be tuples or vectors"))
        if first(entry) isa AbstractMatrix
            length(entry) == 3 ||
                throw(ArgumentError("dynamic matrix entries are [matrix, f, f_args]"))
            matrix, function_value, arguments = entry
        else
            length(entry) == 4 ||
                throw(ArgumentError("dynamic operator entries are [op, couplings, f, f_args]"))
            op, couplings, function_value, arguments = entry
        end
        arguments_tuple = Tuple(arguments)
        entry_format = if dynamic_fmt isa AbstractDict
            requested = get(
                dynamic_fmt,
                (function_value, arguments_tuple),
                get(dynamic_fmt, function_value, dynamic_default),
            )
            _normalize_matrix_format(requested; default=dynamic_default)
        else
            dynamic_default
        end
        dynamic_matrix = if first(entry) isa AbstractMatrix
            _matrix_with_format(matrix, dtype, entry_format; copy_data=copy)
        else
            _assemble(
                selected_basis,
                [OperatorTerm(String(op), collect(couplings))],
                dtype,
                entry_format,
            )
        end
        size(dynamic_matrix) == size(static_matrix) ||
            throw(DimensionMismatch("dynamic matrix has the wrong shape"))
        push!(dynamic_terms, (dynamic_matrix, function_value, arguments_tuple))
    end
    check_herm &&
        !Basis.check_hermitian(selected_basis, static_list, dynamic_list) &&
        throw(ArgumentError("operator list is not Hermitian"))
    check_pcon &&
        !Basis.check_pcon(selected_basis, static_list, dynamic_list) &&
        throw(ArgumentError("operator list violates the selected particle sector"))
    check_symm &&
        !Basis.check_symm(selected_basis, static_list, dynamic_list) &&
        throw(ArgumentError("operator list violates the selected symmetry sector"))
    return Hamiltonian{dtype}(
        selected_basis,
        terms,
        static_matrix,
        dynamic_terms,
        check_herm ? true : nothing,
    )
end

function _matrix_at(H::Hamiltonian, time)
    isempty(H.dynamic_terms) && return H.data
    result = copy(H.data)
    for (matrix, function_value, arguments) in H.dynamic_terms
        result = result + function_value(time, arguments...) * matrix
    end
    return result
end

Base.size(H::Hamiltonian) = size(H.data)
Base.size(H::Hamiltonian, dimension::Integer) = size(H.data, dimension)
Base.eltype(H::Hamiltonian) = eltype(H.data)
Base.getindex(H::Hamiltonian, indices...) = getindex(H.data, indices...)
Base.Matrix(H::Hamiltonian) = Matrix(H.data)
Base.:*(H::Hamiltonian, vector::AbstractVecOrMat) = H.data * vector
LinearAlgebra.mul!(
    output::AbstractVecOrMat,
    H::Hamiltonian,
    value::AbstractVecOrMat,
) = mul!(output, H.data, value)
LinearAlgebra.mul!(
    output::AbstractVecOrMat,
    H::Hamiltonian,
    value::AbstractVecOrMat,
    alpha::Number,
    beta::Number,
) = mul!(output, H.data, value, alpha, beta)
function LinearAlgebra.ishermitian(H::Hamiltonian)
    if H.hermitian === nothing
        H.hermitian = ishermitian(H.data)
    end
    return H.hermitian::Bool
end
LinearAlgebra.issymmetric(H::Hamiltonian) =
    eltype(H) <: Real && ishermitian(H)

function Base.getproperty(H::Hamiltonian, name::Symbol)
    name === :H && return adjoint(H)
    name === :T && return transpose(H)
    name === :Ns && return size(getfield(H, :data), 1)
    name === :dtype && return eltype(getfield(H, :data))
    name === :get_shape && return size(getfield(H, :data))
    name === :is_dense && return (
        getfield(H, :data) isa Matrix ||
        any(first(term) isa Matrix for term in getfield(H, :dynamic_terms))
    )
    name === :nbytes && return Base.summarysize(getfield(H, :data)) +
        sum(
            (Base.summarysize(first(term)) for term in getfield(H, :dynamic_terms));
            init=0,
        )
    name === :ndim && return 2
    name === :shape && return size(getfield(H, :data))
    name === :static && return copy(getfield(H, :data))
    name === :dynamic && return copy(getfield(H, :dynamic_terms))
    return getfield(H, name)
end

function LinearAlgebra.eigvals(H::Hamiltonian)
    matrix = Matrix(H.data)
    return ishermitian(H) ? eigvals(Hermitian(matrix)) : eigvals(matrix)
end

function _transform_hamiltonian(H::Hamiltonian, transform, function_transform=identity)
    transformed_static = transform(H.data)
    static_matrix = _matrix_with_format(
        transformed_static,
        eltype(transformed_static),
        _storage_format(H.data),
    )
    T = eltype(static_matrix)
    dynamic = Tuple{NativeMatrix{T},Any,Tuple}[
        (
            _matrix_with_format(
                transform(matrix),
                T,
                _storage_format(matrix),
            ),
            function_transform(function_value),
            arguments,
        )
        for (matrix, function_value, arguments) in H.dynamic_terms
    ]
    transformed_hermitian =
        H.hermitian === true && transform in (copy, transpose, conj, adjoint) ?
        true : nothing
    return Hamiltonian{T}(
        H.basis,
        copy(H.terms),
        static_matrix,
        dynamic,
        transformed_hermitian,
    )
end

Base.copy(H::Hamiltonian) = _transform_hamiltonian(H, copy)
Base.transpose(H::Hamiltonian) = _transform_hamiltonian(H, transpose)
Base.conj(H::Hamiltonian) = _transform_hamiltonian(
    H,
    conj,
    function_value -> (time, arguments...) -> conj(function_value(time, arguments...)),
)
Base.adjoint(H::Hamiltonian) = _transform_hamiltonian(
    H,
    adjoint,
    function_value -> (time, arguments...) -> conj(function_value(time, arguments...)),
)
ishamiltonian(value) = value isa Hamiltonian

function _hamiltonian_with_formats(
    H::Hamiltonian,
    static_fmt,
    dynamic_fmt;
    copy::Bool=false,
)
    static_format = _normalize_matrix_format(static_fmt)
    dynamic_default = dynamic_fmt isa AbstractDict || dynamic_fmt === nothing ?
        static_format :
        _normalize_matrix_format(dynamic_fmt; default=static_format)
    unchanged = _storage_format(H.data) === static_format
    converted_dynamic = Tuple{NativeMatrix{eltype(H)},Any,Tuple}[]
    for (matrix, function_value, arguments) in H.dynamic_terms
        entry_format = if dynamic_fmt isa AbstractDict
            requested = get(
                dynamic_fmt,
                (function_value, arguments),
                get(dynamic_fmt, function_value, dynamic_default),
            )
            _normalize_matrix_format(requested; default=dynamic_default)
        else
            dynamic_default
        end
        unchanged &= _storage_format(matrix) === entry_format
        push!(
            converted_dynamic,
            (
                _matrix_with_format(
                    matrix,
                    eltype(H),
                    entry_format;
                    copy_data=copy,
                ),
                function_value,
                arguments,
            ),
        )
    end
    unchanged && !copy && return H
    static_matrix = _matrix_with_format(
        H.data,
        eltype(H),
        static_format;
        copy_data=copy,
    )
    return Hamiltonian{eltype(H)}(
        H.basis,
        Base.copy(H.terms),
        static_matrix,
        converted_dynamic,
        H.hermitian,
    )
end

as_dense_format(H::Hamiltonian; copy::Bool=false) =
    _hamiltonian_with_formats(H, :dense, :dense; copy)
function as_sparse_format(
    H::Hamiltonian;
    static_fmt=:csc,
    dynamic_fmt=nothing,
    copy::Bool=false,
)
    return _hamiltonian_with_formats(H, static_fmt, dynamic_fmt; copy)
end
struct ActionLinearOperator{T,F} <: AbstractMatrix{T}
    n::Int
    action!::F
    hermitian::Bool
end

Base.size(operator::ActionLinearOperator) = (operator.n, operator.n)
Base.size(operator::ActionLinearOperator, dimension::Integer) =
    size(operator)[dimension]
Base.eltype(::Type{ActionLinearOperator{T,F}}) where {T,F} = T
Base.eltype(operator::ActionLinearOperator) = eltype(typeof(operator))
LinearAlgebra.ishermitian(operator::ActionLinearOperator) = operator.hermitian
LinearAlgebra.issymmetric(operator::ActionLinearOperator) =
    eltype(operator) <: Real && operator.hermitian
function LinearAlgebra.mul!(
    output::AbstractVector,
    operator::ActionLinearOperator,
    value::AbstractVector,
    alpha::Number=true,
    beta::Number=false,
)
    if isone(alpha) && iszero(beta)
        operator.action!(output, value)
    else
        temporary = similar(
            output,
            promote_type(eltype(operator), eltype(value)),
        )
        operator.action!(temporary, value)
        iszero(beta) ? fill!(output, zero(eltype(output))) : lmul!(beta, output)
        axpy!(alpha, temporary, output)
    end
    return output
end
function LinearAlgebra.mul!(
    output::AbstractMatrix,
    operator::ActionLinearOperator,
    value::AbstractMatrix,
    alpha::Number=true,
    beta::Number=false,
)
    iszero(beta) ? fill!(output, zero(eltype(output))) : lmul!(beta, output)
    for column in axes(value, 2)
        mul!(
            @view(output[:, column]),
            operator,
            @view(value[:, column]),
            alpha,
            true,
        )
    end
    return output
end
function Base.:*(
    operator::ActionLinearOperator{T},
    value::AbstractVector{S},
) where {T,S}
    output = zeros(promote_type(T, S), size(operator, 1))
    return mul!(output, operator, value)
end
function Base.:*(
    operator::ActionLinearOperator{T},
    value::AbstractMatrix{S},
) where {T,S}
    output = zeros(
        promote_type(T, S),
        size(operator, 1),
        size(value, 2),
    )
    return mul!(output, operator, value)
end
function Base.getindex(operator::ActionLinearOperator{T}, row::Int, column::Int) where {T}
    input = zeros(T, operator.n)
    input[column] = one(T)
    output = operator * input
    return output[row]
end

function aslinearoperator(H::Hamiltonian; time=0)
    isempty(H.dynamic_terms) && iszero(time) && return H
    T = promote_type(
        eltype(H),
        (
            typeof(function_value(time, arguments...))
            for (_, function_value, arguments) in H.dynamic_terms
        )...,
    )
    action! = (output, input) ->
        apply(H, input; time, out=output, overwrite_out=true)
    hermitian = H.hermitian === true && all(
        isreal(function_value(time, arguments...))
        for (_, function_value, arguments) in H.dynamic_terms
    )
    return ActionLinearOperator{T,typeof(action!)}(size(H, 1), action!, hermitian)
end
check_is_dense(H::Hamiltonian) = H.is_dense
function astype(
    H::Hamiltonian,
    ::Type{T};
    copy::Bool=false,
    kwargs...,
) where {T<:Number}
    dynamic = Tuple{NativeMatrix{T},Any,Tuple}[
        (
            _matrix_with_format(matrix, T, _storage_format(matrix)),
            function_value,
            arguments,
        )
        for (matrix, function_value, arguments) in H.dynamic_terms
    ]
    static_matrix = _matrix_with_format(H.data, T, _storage_format(H.data))
    return Hamiltonian{T}(
        H.basis,
        Base.copy(H.terms),
        static_matrix,
        dynamic,
        H.hermitian,
    )
end
function diagonal(H::Hamiltonian; time=0)
    result = collect(diag(H.data))
    for (matrix, function_value, arguments) in H.dynamic_terms
        result .+= function_value(time, arguments...) .* diag(matrix)
    end
    return result
end
toarray(H::Hamiltonian; time=0, order=nothing, out=nothing) =
    _copy_or_write(Matrix(_matrix_at(H, time)), out)
todense(H::Hamiltonian; kwargs...) = toarray(H; kwargs...)
tocsc(H::Hamiltonian; time=0) = SparseMatrixCSC(_matrix_at(H, time))
tocsr(H::Hamiltonian; time=0) = SparseMatrixCSR(_matrix_at(H, time))

function _copy_or_write(value, out)
    out === nothing && return value
    axes(out) == axes(value) ||
        throw(DimensionMismatch("out must have the same axes as the result"))
    copyto!(out, value)
    return out
end

_action_output(::Type{T}, rows::Integer, value::AbstractVector) where {T} =
    zeros(T, rows)
_action_output(::Type{T}, rows::Integer, value::AbstractMatrix) where {T} =
    zeros(T, rows, size(value, 2))

function _matrix_mul_add!(
    output::AbstractVecOrMat,
    matrix,
    value::AbstractVecOrMat,
    alpha::Number,
    beta::Number,
)
    if applicable(mul!, output, matrix, value, alpha, beta)
        mul!(output, matrix, value, alpha, beta)
        return output
    end
    product = matrix * value
    if iszero(beta)
        @. output = alpha * product
    else
        @. output = beta * output + alpha * product
    end
    return output
end

function apply(
    H::Hamiltonian,
    value::AbstractVecOrMat;
    time=0,
    check::Bool=true,
    out=nothing,
    overwrite_out::Bool=true,
    a::Number=1,
)
    size(value, 1) == size(H, 2) ||
        throw(DimensionMismatch("Hamiltonian and value dimensions do not match"))
    T = promote_type(
        eltype(H),
        eltype(value),
        typeof(a),
    )
    result = out === nothing ?
        _action_output(T, size(H, 1), value) :
        out
    expected_axes = value isa AbstractVector ?
        (axes(H.data, 1),) :
        (axes(H.data, 1), axes(value, 2))
    axes(result) == expected_axes ||
        throw(DimensionMismatch("out must have the same axes as the result"))
    input = Base.mightalias(result, value) ? copy(value) : value
    beta = out === nothing || overwrite_out ? zero(a) : one(a)
    _matrix_mul_add!(result, H.data, input, a, beta)
    for (matrix, function_value, arguments) in H.dynamic_terms
        coefficient = function_value(time, arguments...)
        iszero(coefficient) && continue
        required_type = promote_type(eltype(result), typeof(a * coefficient))
        if out === nothing && required_type !== eltype(result)
            widened = _action_output(
                required_type,
                size(H, 1),
                value,
            )
            copyto!(widened, result)
            result = widened
        end
        _matrix_mul_add!(result, matrix, input, a * coefficient, true)
    end
    return result
end

function right_apply(
    H::Hamiltonian,
    value;
    time=0,
    check::Bool=true,
    out=nothing,
    overwrite_out::Bool=true,
    a::Number=1,
)
    size(value, ndims(value)) == size(H, 1) ||
        throw(DimensionMismatch("value and Hamiltonian dimensions do not match"))
    T = promote_type(eltype(H), eltype(value), typeof(a))
    result = out === nothing ? similar(value, T) : out
    input = Base.mightalias(result, value) ? copy(value) : value
    beta = out === nothing || overwrite_out ? zero(a) : one(a)
    _right_matrix_mul_add!(result, input, H.data, a, beta)
    for (matrix, function_value, arguments) in H.dynamic_terms
        coefficient = function_value(time, arguments...)
        iszero(coefficient) && continue
        _right_matrix_mul_add!(result, input, matrix, a * coefficient, true)
    end
    return result
end

function _right_matrix_mul_add!(output, value, matrix, alpha, beta)
    if value isa AbstractVector
        return _matrix_mul_add!(
            output,
            transpose(matrix),
            value,
            alpha,
            beta,
        )
    end
    if applicable(
        mul!,
        transpose(output),
        transpose(matrix),
        transpose(value),
        alpha,
        beta,
    )
        mul!(
            transpose(output),
            transpose(matrix),
            transpose(value),
            alpha,
            beta,
        )
        return output
    end
    product = value * matrix
    iszero(beta) ? copyto!(output, alpha .* product) :
        (@. output = beta * output + alpha * product)
    return output
end

function eigh(H::Hamiltonian; time=0, kwargs...)
    matrix = Matrix(_matrix_at(H, time))
    decomposition = ishermitian(matrix) ?
        eigen(Hermitian(matrix)) :
        eigen(matrix)
    return decomposition.values, decomposition.vectors
end

function eigsh(
    H::Hamiltonian;
    time=0,
    k::Integer=min(6, size(H, 1)),
    which=:SA,
    return_eigenvectors::Bool=true,
    kwargs...,
)
    sigma = get(kwargs, :sigma, nothing)
    source = if sigma === nothing
        aslinearoperator(H; time)
    else
        matrix = _matrix_at(H, time)
        matrix isa Union{SparseMatrixCSR,DIAMatrix} &&
            (matrix = sparse(matrix))
        hermitian_at_time = H.hermitian === true && all(
            isreal(function_value(time, arguments...))
            for (_, function_value, arguments) in H.dynamic_terms
        )
        hermitian_at_time ? Hermitian(matrix) : matrix
    end
    n = size(source, 1)
    1 <= k <= n || throw(ArgumentError("k must lie in 1:Ns"))
    requested = Symbol(uppercase(String(which)))
    requested in (:SA, :LA, :SM, :LM, :BE) ||
        throw(ArgumentError("which must be SA, LA, SM, LM, or BE"))
    if k >= n - 1
        values, vectors = eigh(H; time)
        indices = requested === :SA ?
            sortperm(real.(values))[1:k] :
            requested === :LA ?
            sortperm(real.(values); rev=true)[1:k] :
            requested === :SM ?
            sortperm(abs.(values))[1:k] :
            requested === :LM ?
            sortperm(abs.(values); rev=true)[1:k] :
            vcat(
                sortperm(real.(values))[1:fld(k, 2)],
                sortperm(real.(values); rev=true)[1:cld(k, 2)],
            )
        selected_values = values[indices]
        return return_eigenvectors ?
            (selected_values, vectors[:, indices]) :
            selected_values
    end
    arpack_which = requested === :SA ? :SR : requested === :LA ? :LR : requested
    arpack_result = Arpack.eigs(
        source;
        nev=k,
        which=arpack_which,
        ritzvec=return_eigenvectors,
        kwargs...,
    )
    values = arpack_result[1]
    vectors = return_eigenvectors ? arpack_result[2] : nothing
    order = requested === :SA ?
        sortperm(real.(values)) :
        requested === :LA ?
        sortperm(real.(values); rev=true) :
        requested === :SM ?
        sortperm(abs.(values)) :
        requested === :LM ?
        sortperm(abs.(values); rev=true) :
        eachindex(values)
    ordered_values = values[order]
    ishermitian(source) && (ordered_values = real.(ordered_values))
    return return_eigenvectors ?
        (ordered_values, vectors[:, order]) :
        ordered_values
end

function _operator_mul!(
    output::AbstractVector,
    operator,
    input::AbstractVector,
)
    if applicable(mul!, output, operator, input)
        mul!(output, operator, input)
    else
        copyto!(output, operator * input)
    end
    return output
end

function _krylov_factorization(
    operator,
    vector::AbstractVector,
    ::Type{T};
    krylov_dim::Integer=30,
) where {T}
    size(operator, 1) == size(operator, 2) == length(vector) ||
        throw(DimensionMismatch("operator and vector dimensions do not match"))
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive"))
    initial = Vector{T}(vector)
    beta = norm(initial)

    dimension = length(initial)
    steps = min(Int(krylov_dim), dimension)
    basis = zeros(T, dimension, steps + 1)
    hessenberg = zeros(T, steps + 1, steps)
    @views @. basis[:, 1] = initial / beta
    residual = similar(initial)
    completed = steps
    terminal_norm = 0.0
    hermitian = ishermitian(operator)
    if hermitian
        previous_norm = zero(real(float(one(T))))
        for column in 1:steps
            _operator_mul!(residual, operator, @view(basis[:, column]))
            column > 1 &&
                axpy!(-previous_norm, @view(basis[:, column - 1]), residual)
            coefficient = dot(@view(basis[:, column]), residual)
            hessenberg[column, column] = coefficient
            axpy!(-coefficient, @view(basis[:, column]), residual)
            terminal_norm = norm(residual)
            hessenberg[column + 1, column] = terminal_norm
            column < steps &&
                (hessenberg[column, column + 1] = terminal_norm)
            if terminal_norm <= 100eps(Float64) * max(1.0, norm(hessenberg))
                completed = column
                break
            elseif column < steps
                @views @. basis[:, column + 1] = residual / terminal_norm
            end
            previous_norm = terminal_norm
        end
    else
        for column in 1:steps
            _operator_mul!(residual, operator, @view(basis[:, column]))
            for row in 1:column
                coefficient = dot(@view(basis[:, row]), residual)
                hessenberg[row, column] = coefficient
                axpy!(-coefficient, @view(basis[:, row]), residual)
            end
            for row in 1:column
                correction = dot(@view(basis[:, row]), residual)
                hessenberg[row, column] += correction
                axpy!(-correction, @view(basis[:, row]), residual)
            end
            terminal_norm = norm(residual)
            hessenberg[column + 1, column] = terminal_norm
            if terminal_norm <= 100eps(Float64) * max(1.0, norm(hessenberg))
                completed = column
                break
            elseif column < steps
                @views @. basis[:, column + 1] = residual / terminal_norm
            end
        end
    end
    return (
        initial=initial,
        beta=beta,
        basis=basis,
        hessenberg=hessenberg,
        completed=completed,
        terminal_norm=terminal_norm,
        dimension=dimension,
        steps=steps,
        hermitian=hermitian,
    )
end

function _krylov_project(factorization, scale::Number)
    completed = factorization.completed
    reduced = Matrix(@view factorization.hessenberg[1:completed, 1:completed])
    exponential = exp(scale .* reduced)
    result = similar(
        factorization.initial,
        promote_type(eltype(factorization.initial), eltype(exponential)),
    )
    mul!(
        result,
        @view(factorization.basis[:, 1:completed]),
        @view(exponential[:, 1]),
    )
    lmul!(factorization.beta, result)
    estimate =
        completed == factorization.steps &&
        completed < factorization.dimension ?
        abs(scale) *
        factorization.terminal_norm *
        factorization.beta *
        abs(exponential[end, 1]) :
        0.0
    return result, estimate
end

function _krylov_expmv_vector(
    operator,
    vector::AbstractVector,
    scale::Number;
    tol::Real=1e-12,
    krylov_dim::Integer=30,
    depth::Integer=0,
)
    size(operator, 1) == size(operator, 2) == length(vector) ||
        throw(DimensionMismatch("operator and vector dimensions do not match"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive"))
    T = promote_type(eltype(operator), eltype(vector), typeof(scale))
    initial = Vector{T}(vector)
    iszero(scale) && return initial
    iszero(norm(initial)) && return initial
    factorization = _krylov_factorization(
        operator,
        initial,
        T;
        krylov_dim,
    )
    result, estimate = _krylov_project(factorization, scale)
    if estimate > tol * max(1.0, norm(result))
        depth < 14 || throw(ErrorException(
            "Krylov exponential action did not converge; increase krylov_dim",
        ))
        midpoint = _krylov_expmv_vector(
            operator,
            initial,
            scale / 2;
            tol=tol / 2,
            krylov_dim,
            depth=depth + 1,
        )
        return _krylov_expmv_vector(
            operator,
            midpoint,
            scale / 2;
            tol=tol / 2,
            krylov_dim,
            depth=depth + 1,
        )
    end
    return result
end

function _krylov_expmv_times(
    operator,
    vector::AbstractVector,
    scales::AbstractVector;
    tol::Real=1e-12,
    krylov_dim::Integer=30,
)
    size(operator, 1) == size(operator, 2) == length(vector) ||
        throw(DimensionMismatch("operator and vector dimensions do not match"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    krylov_dim > 0 || throw(ArgumentError("krylov_dim must be positive"))
    T = promote_type(eltype(operator), eltype(vector), eltype(scales))
    initial = Vector{T}(vector)
    results = Matrix{T}(undef, length(initial), length(scales))
    isempty(scales) && return results
    if iszero(norm(initial)) || all(iszero, scales)
        for column in axes(results, 2)
            copyto!(@view(results[:, column]), initial)
        end
        return results
    end
    factorization = _krylov_factorization(
        operator,
        initial,
        T;
        krylov_dim,
    )
    completed = factorization.completed
    if factorization.hermitian
        reduced = Hermitian(
            Matrix(@view factorization.hessenberg[1:completed, 1:completed]),
        )
        decomposition = eigen(reduced)
        first_coordinates = adjoint(decomposition.vectors)[:, 1]
        reduced_state = similar(initial)
        for (column, scale) in pairs(scales)
            if iszero(scale)
                copyto!(@view(results[:, column]), initial)
                continue
            end
            coefficients =
                decomposition.vectors *
                (exp.(scale .* decomposition.values) .* first_coordinates)
            mul!(
                reduced_state,
                @view(factorization.basis[:, 1:completed]),
                coefficients,
            )
            state = factorization.beta .* reduced_state
            estimate =
                completed == factorization.steps &&
                completed < factorization.dimension ?
                abs(scale) *
                factorization.terminal_norm *
                factorization.beta *
                abs(coefficients[end]) :
                0.0
            if estimate > tol * max(1.0, norm(state))
                state = _krylov_expmv_vector(
                    operator,
                    initial,
                    scale;
                    tol,
                    krylov_dim,
                )
            end
            copyto!(@view(results[:, column]), state)
        end
    else
        for (column, scale) in pairs(scales)
            if iszero(scale)
                copyto!(@view(results[:, column]), initial)
                continue
            end
            state, estimate = _krylov_project(factorization, scale)
            if estimate > tol * max(1.0, norm(state))
                state = _krylov_expmv_vector(
                    operator,
                    initial,
                    scale;
                    tol,
                    krylov_dim,
                )
            end
            copyto!(@view(results[:, column]), state)
        end
    end
    return results
end

_krylov_expmv(operator, value::AbstractVector, scale::Number; kwargs...) =
    _krylov_expmv_vector(operator, value, scale; kwargs...)

function _krylov_expmv(
    operator,
    value::AbstractMatrix,
    scale::Number;
    kwargs...,
)
    size(value, 1) == size(operator, 2) ||
        throw(DimensionMismatch("operator and matrix dimensions do not match"))
    T = promote_type(eltype(operator), eltype(value), typeof(scale))
    result = Matrix{T}(
        undef,
        size(value, 1),
        size(value, 2),
    )
    for column in axes(value, 2)
        state = _krylov_expmv_vector(
            operator,
            @view(value[:, column]),
            scale;
            kwargs...,
        )
        copyto!(@view(result[:, column]), state)
    end
    return result
end

function _evolution_derivative(H, value, time, imag_time)
    left = apply(H, value; time)
    if value isa AbstractVector
        return (imag_time ? -one(eltype(left)) : -im) .* left
    end
    right = right_apply(H, value; time)
    return imag_time ?
        -(left + right) / 2 :
        -im .* (left - right)
end

function _rk45_interval(
    H,
    initial,
    start::Real,
    target::Real,
    initial_step::Real,
    max_step::Real,
    imag_time::Bool;
    rtol::Real=1e-9,
    atol::Real=1e-11,
)
    state = initial
    time = float(start)
    step = min(max_step, max(initial_step, eps(float(target + one(target)))))
    while time < target
        step = min(step, target - time)
        step > eps(max(abs(time), 1.0)) ||
            throw(ErrorException("adaptive evolution step underflow"))
        k1 = _evolution_derivative(H, state, time, imag_time)
        k2 = _evolution_derivative(
            H,
            state + step * (1 / 5) * k1,
            time + step * (1 / 5),
            imag_time,
        )
        k3 = _evolution_derivative(
            H,
            state + step * ((3 / 40) * k1 + (9 / 40) * k2),
            time + step * (3 / 10),
            imag_time,
        )
        k4 = _evolution_derivative(
            H,
            state +
            step *
            ((44 / 45) * k1 - (56 / 15) * k2 + (32 / 9) * k3),
            time + step * (4 / 5),
            imag_time,
        )
        k5 = _evolution_derivative(
            H,
            state +
            step *
            (
                (19372 / 6561) * k1 -
                (25360 / 2187) * k2 +
                (64448 / 6561) * k3 -
                (212 / 729) * k4
            ),
            time + step * (8 / 9),
            imag_time,
        )
        k6 = _evolution_derivative(
            H,
            state +
            step *
            (
                (9017 / 3168) * k1 -
                (355 / 33) * k2 +
                (46732 / 5247) * k3 +
                (49 / 176) * k4 -
                (5103 / 18656) * k5
            ),
            time + step,
            imag_time,
        )
        fifth = state +
            step *
            (
                (35 / 384) * k1 +
                (500 / 1113) * k3 +
                (125 / 192) * k4 -
                (2187 / 6784) * k5 +
                (11 / 84) * k6
            )
        k7 = _evolution_derivative(H, fifth, time + step, imag_time)
        fourth = state +
            step *
            (
                (5179 / 57600) * k1 +
                (7571 / 16695) * k3 +
                (393 / 640) * k4 -
                (92097 / 339200) * k5 +
                (187 / 2100) * k6 +
                (1 / 40) * k7
            )
        scale = @. atol + rtol * max(abs(state), abs(fifth))
        error = maximum(abs.(fifth .- fourth) ./ scale; init=0.0)
        if error <= 1
            state = fifth
            time += step
        end
        factor = error == 0 ? 5.0 : clamp(0.9 * error^(-1 / 5), 0.2, 5.0)
        step = min(max_step, step * factor)
    end
    return state, step
end

function evolve(
    H::Hamiltonian,
    v0::AbstractVecOrMat,
    t0::Real,
    times;
    eom=:SE,
    iterate::Bool=false,
    imag_time::Bool=false,
    max_step::Real=0.01,
    kwargs...,
)
    if !isempty(H.dynamic_terms)
        max_step > 0 || throw(ArgumentError("max_step must be positive"))
        state = ComplexF64.(v0)
        current = float(t0)
        states = Any[]
        next_step = max_step
        rtol = get(kwargs, :rtol, get(kwargs, :tol, 1e-9))
        atol = get(kwargs, :atol, 1e-11)
        for target in times
            target >= current ||
                throw(ArgumentError("times must be sorted and not precede t0"))
            state, next_step = _rk45_interval(
                H,
                state,
                current,
                float(target),
                next_step,
                max_step,
                imag_time;
                rtol,
                atol,
            )
            current = float(target)
            if imag_time
                state = state isa AbstractVector ?
                    state / norm(state) :
                    state / tr(state)
            end
            push!(states, copy(state))
        end
        iterate && return (state for state in states)
        return v0 isa AbstractVector ?
            reduce(hcat, states) :
            cat(states...; dims=3)
    end

    offsets = collect(times) .- t0
    if v0 isa AbstractVector
        scales = imag_time ? -offsets : -im .* offsets
        states = _krylov_expmv_times(
            H.data,
            v0,
            scales;
            tol=get(kwargs, :tol, 1e-12),
            krylov_dim=get(kwargs, :krylov_dim, 30),
        )
        if imag_time
            for state in eachcol(states)
                rmul!(state, inv(norm(state)))
            end
        end
        iterate && return (copy(state) for state in eachcol(states))
        return states
    end
    size(v0, 1) == size(v0, 2) == size(H, 1) ||
        throw(DimensionMismatch("density matrix must match Hamiltonian"))
    states = [
        begin
            left = _krylov_expmv(
                H.data,
                v0,
                -im * time;
                tol=get(kwargs, :tol, 1e-12),
                krylov_dim=get(kwargs, :krylov_dim, 30),
            )
            adjoint(_krylov_expmv(
                H.data,
                adjoint(left),
                -im * time;
                tol=get(kwargs, :tol, 1e-12),
                krylov_dim=get(kwargs, :krylov_dim, 30),
            ))
        end
        for time in offsets
    ]
    iterate && return (state for state in states)
    return cat(states...; dims=3)
end

function _trace_product(left::AbstractMatrix, right::AbstractMatrix)
    size(left, 2) == size(right, 1) &&
        size(left, 1) == size(right, 2) ||
        throw(DimensionMismatch("trace product requires transposed dimensions"))
    T = promote_type(eltype(left), eltype(right))
    result = zero(T)
    @inbounds for column in axes(left, 2), row in axes(left, 1)
        result += left[row, column] * right[column, row]
    end
    return result
end

function expt_value(H::Hamiltonian, state; time=0, enforce_pure::Bool=false, kwargs...)
    if state isa AbstractVector
        return dot(state, apply(H, state; time))
    elseif ndims(state) == 3
        return [
            expt_value(H, @view(state[:, :, index]); time)
            for index in axes(state, 3)
        ]
    elseif enforce_pure || size(state, 2) != size(H, 1)
        acted = apply(H, state; time)
        return [
            dot(@view(state[:, column]), @view(acted[:, column]))
            for column in axes(state, 2)
        ]
    end
    result = _trace_product(state, H.data)
    for (matrix, function_value, arguments) in H.dynamic_terms
        result +=
            function_value(time, arguments...) * _trace_product(state, matrix)
    end
    return result
end

function matrix_ele(
    H::Hamiltonian,
    left,
    right;
    time=0,
    diagonal::Bool=false,
    check::Bool=true,
)
    elements = left' * apply(H, right; time)
    return diagonal && elements isa AbstractMatrix ? diag(elements) : elements
end

function project_to(H::Hamiltonian, projector)
    P = projector isa AbstractMatrix ? projector : Matrix(projector)
    projected = if size(H, 1) == size(P, 1)
        P' * H.data * P
    elseif size(H, 1) == size(P, 2)
        P * H.data * P'
    else
        throw(DimensionMismatch("Hamiltonian and projector dimensions do not match"))
    end
    size(projected) == size(H.data) ||
        return projected
    return _hamiltonian_from_data(H.basis, projected)
end

function quant_fluct(H::Hamiltonian, state; time=0, enforce_pure::Bool=false, kwargs...)
    mean = expt_value(H, state; time, enforce_pure)
    if state isa AbstractVector
        first_action = apply(H, state; time)
        second = dot(state, apply(H, first_action; time))
        return second - mean^2
    elseif enforce_pure || (ndims(state) == 2 && size(state, 2) != size(H, 1))
        first_action = apply(H, state; time)
        second_action = apply(H, first_action; time)
        second = [
            dot(@view(state[:, column]), @view(second_action[:, column]))
            for column in axes(state, 2)
        ]
        return second .- mean .^ 2
    end
    first_action = apply(H, state; time)
    second_action = right_apply(H, first_action; time)
    second = tr(second_action)
    return second .- mean .^ 2
end

function rotate_by(H::Hamiltonian, other; generator::Bool=false, kwargs...)
    U = generator ?
        get_mat(ExpOp(_operator_dense_or_self(other); kwargs...)) :
        other isa ExpOp ? get_mat(other) : _operator_dense_or_self(other)
    return _hamiltonian_from_data(H.basis, U' * H.data * U)
end

LinearAlgebra.tr(H::Hamiltonian) = tr(H.data)
function update_matrix_formats!(
    H::Hamiltonian,
    static_fmt,
    dynamic_fmt=nothing,
)
    converted = _hamiltonian_with_formats(
        H,
        static_fmt,
        dynamic_fmt;
        copy=false,
    )
    converted === H && return H
    H.data = converted.data
    H.dynamic_terms = converted.dynamic_terms
    return H
end

"""
    QuantumLinearOperator(basis, static_list; diagonal=nothing)

Matrix-free operator backed directly by local basis actions. Construction does
not assemble a dense or sparse matrix; explicit materialization happens only
when `Matrix(operator)` is requested.
"""
mutable struct QuantumLinearOperator{
    T<:Number,
    B<:AbstractBasis,
    SL<:AbstractVector{<:OperatorTerm},
    AL<:Tuple,
} <: AbstractMatrix{T}
    basis::B
    static_list::SL
    action_terms::AL
    explicit_data::Union{Nothing,Matrix{T}}
    diagonal::Vector{T}
    extracted_diagonal::Vector{T}
    hermitian::Bool
    transposed::Bool
    conjugated::Bool
end

_is_structurally_diagonal(op::AbstractString) =
    all(character -> character in ('I', 'z', 'n', '|'), op)

function _extract_diagonal_terms(basis, terms)
    remaining = OperatorTerm[]
    diagonal = zeros(ComplexF64, length(basis))
    for term in terms
        if _is_structurally_diagonal(term.op)
            term_matrix = sparse(
                operator_matrix(
                    basis,
                    term.op,
                    term.couplings;
                    sparse=true,
                ),
            )
            diagonal .+= diag(term_matrix)
        else
            push!(remaining, term)
        end
    end
    return remaining, diagonal
end

function _project_parent_diagonal(basis, parent_diagonal)
    Basis._has_symmetry(basis.symmetry) || return parent_diagonal
    projector = basis.symmetry.projector
    result = zeros(eltype(parent_diagonal), size(projector, 2))
    rows = rowvals(projector)
    values = nonzeros(projector)
    @inbounds for column in axes(projector, 2)
        for pointer in nzrange(projector, column)
            result[column] +=
                abs2(values[pointer]) * parent_diagonal[rows[pointer]]
        end
    end
    return result
end

function _accumulate_spin_diagonal!(
    diagonal,
    parent::SpinBasis1D,
    term::OperatorTerm{C},
) where {C}
    operator_length = length(term.op)
    for coupling in term.couplings
        sites = Base.tail(coupling)
        @inbounds for (column, initial_state) in pairs(parent.encoded_states)
            amplitude = complex(first(coupling))
            state = initial_state
            alive = true
            for operator_index in operator_length:-1:1
                state, factor, alive = _apply_local(
                    parent,
                    state,
                    term.op[operator_index],
                    sites[operator_index],
                )
                alive || break
                amplitude *= factor
            end
            alive && (diagonal[column] += amplitude)
        end
    end
    return diagonal
end

function _extract_diagonal_terms(basis::SpinBasis1D, terms)
    remaining = OperatorTerm[]
    parent = Basis._has_symmetry(basis.symmetry) ?
        Basis._parent_basis_for_checks(basis) :
        basis
    diagonal = zeros(ComplexF64, length(parent))
    for term in terms
        if !_is_structurally_diagonal(term.op)
            push!(remaining, term)
            continue
        end
        _accumulate_spin_diagonal!(diagonal, parent, term)
    end
    return remaining, _project_parent_diagonal(basis, diagonal)
end

function _accumulate_discrete_diagonal!(
    diagonal,
    parent::Basis.DiscreteBasis,
    term::OperatorTerm{C},
    weights,
) where {C}
    for coupling in term.couplings
        actions = Basis._operator_actions(
            parent,
            term.op,
            coupling[2:end],
        )
        @inbounds for (column, initial_state) in
                      pairs(parent.encoded_states)
            amplitude = complex(first(coupling))
            encoded = initial_state
            alive = true
            for (op, site, species) in Iterators.reverse(actions)
                encoded, factor, alive =
                    Basis._apply_discrete_encoded_local(
                        parent,
                        encoded,
                        op,
                        site,
                        species,
                        weights[site],
                        weights,
                    )
                alive || break
                amplitude *= factor
            end
            alive && (diagonal[column] += amplitude)
        end
    end
    return diagonal
end

function _extract_diagonal_terms(basis::Basis.DiscreteBasis, terms)
    remaining = OperatorTerm[]
    parent = Basis._has_symmetry(basis.symmetry) ?
        Basis._parent_basis_for_checks(basis) :
        basis
    diagonal = zeros(ComplexF64, length(parent))
    weights = UInt64[
        UInt64(parent.sps)^(site - 1) for site in 1:parent.L
    ]
    for term in terms
        if !_is_structurally_diagonal(term.op)
            push!(remaining, term)
            continue
        end
        _accumulate_discrete_diagonal!(
            diagonal,
            parent,
            term,
            weights,
        )
    end
    return remaining, _project_parent_diagonal(basis, diagonal)
end

function QuantumLinearOperator(
    basis::AbstractBasis,
    static_list::AbstractVector{<:OperatorTerm};
    diagonal=nothing,
    check_symm::Bool=true,
    check_herm::Bool=true,
    check_pcon::Bool=true,
)
    public_terms = OperatorTerm[static_list...]
    checked_terms = _consolidate_terms(public_terms)
    normalized, extracted_diagonal = if !check_symm &&
                                        hasfield(typeof(basis), :symmetry) &&
                                        Basis._has_symmetry(
                                            getfield(basis, :symmetry),
                                        )
        checked_terms, zeros(ComplexF64, length(basis))
    else
        _extract_diagonal_terms(basis, checked_terms)
    end
    hermitian =
        basis isa Union{SpinBasis1D,Basis.DiscreteBasis{:boson}} ?
        _structured_is_hermitian(checked_terms) :
        nothing
    hermitian === nothing &&
        (hermitian = _matrixfree_is_hermitian(basis, checked_terms))
    check_herm && !hermitian &&
        throw(ArgumentError("operator list is not Hermitian"))
    check_pcon && !Basis.check_pcon(basis, checked_terms, Any[]) &&
        throw(ArgumentError("operator list violates the selected particle sector"))
    check_symm && !Basis.check_symm(basis, checked_terms, Any[]) &&
        throw(ArgumentError("operator list violates the selected symmetry sector"))
    T = _coefficient_type(checked_terms)
    Basis._basis_requires_complex(basis) &&
        (T = promote_type(T, ComplexF64))
    diagonal !== nothing &&
        (T = promote_type(T, eltype(diagonal)))
    diagonal_values = diagonal === nothing ?
        zeros(T, length(basis)) :
        Vector{T}(diagonal)
    length(diagonal_values) == length(basis) ||
        throw(DimensionMismatch("diagonal must have length Ns"))
    check_herm && !all(isreal, diagonal_values) &&
        throw(ArgumentError("a Hermitian operator requires a real diagonal"))
    action_terms = Tuple(normalized)
    return QuantumLinearOperator{
        T,
        typeof(basis),
        typeof(public_terms),
        typeof(action_terms),
    }(
        basis,
        public_terms,
        action_terms,
        nothing,
        diagonal_values,
        Vector{T}(extracted_diagonal),
        check_herm ? true : hermitian && all(isreal, diagonal_values),
        false,
        false,
    )
end

function _apply_spin_term!(
    result::AbstractVector,
    basis::SpinBasis1D,
    term::OperatorTerm,
    vector::AbstractVector,
    alpha::Number=1,
    transposed::Bool=false,
    conjugated::Bool=false,
)
    for coupling in term.couplings
        coefficient = alpha * first(coupling)
        sites = Base.tail(coupling)
        operator_length = length(term.op)
        for (column, initial_state) in pairs(basis.encoded_states)
            amplitude = coefficient
            state = initial_state
            alive = true
            for operator_index in operator_length:-1:1
                op = term.op[operator_index]
                site = sites[operator_index]
                state, factor, alive = _apply_local(basis, state, op, site)
                alive || break
                amplitude *= factor
            end
            alive || continue
            row = get(basis.lookup, state, 0)
            row == 0 && continue
            transition = conjugated ? conj(amplitude) : amplitude
            source = transposed ? row : column
            destination = transposed ? column : row
            input = vector[source]
            iszero(input) || (result[destination] += transition * input)
        end
    end
    return result
end

function _apply_spin_terms!(
    result::AbstractVector,
    basis::SpinBasis1D,
    terms,
    vector::AbstractVector,
    alpha::Number=1,
    transposed::Bool=false,
    conjugated::Bool=false,
)
    if Basis._has_symmetry(basis.symmetry)
        parent = Basis._parent_basis_for_checks(basis)
        projector = basis.symmetry.projector
        ordinary_projection = transposed == conjugated
        lifted = ordinary_projection ?
            projector * vector :
            conj(projector) * vector
        acted = zeros(
            promote_type(eltype(lifted), _coefficient_type(terms)),
            length(parent),
        )
        _apply_spin_terms!(
            acted,
            parent,
            terms,
            lifted,
            one(alpha),
            transposed,
            conjugated,
        )
        projection = ordinary_projection ? projector' : transpose(projector)
        mul!(result, projection, acted, alpha, true)
        return result
    end
    for term in terms
        _apply_spin_term!(
            result,
            basis,
            term,
            vector,
            alpha,
            transposed,
            conjugated,
        )
    end
    return result
end

function _apply_spin_terms(
    basis::SpinBasis1D,
    terms,
    vector::AbstractVector,
)
    T = promote_type(eltype(vector), _coefficient_type(terms))
    result = zeros(T, length(basis))
    return _apply_spin_terms!(result, basis, terms, vector)
end

function _apply_discrete_term!(
    result::AbstractVector,
    basis::Basis.DiscreteBasis,
    term::OperatorTerm,
    vector::AbstractVector,
    alpha::Number=1,
    transposed::Bool=false,
    conjugated::Bool=false,
)
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.L
    ]
    for coupling in term.couplings
        actions = Basis._operator_actions(
            basis,
            term.op,
            coupling[2:end],
        )
        coefficient = alpha * first(coupling)
        for (column, initial_state) in pairs(basis.encoded_states)
            encoded = initial_state
            amplitude = coefficient
            alive = true
            for (op, site, species) in Iterators.reverse(actions)
                encoded, factor, alive = Basis._apply_discrete_encoded_local(
                    basis,
                    encoded,
                    op,
                    site,
                    species,
                    weights[site],
                    weights,
                )
                alive || break
                amplitude *= factor
            end
            alive || continue
            row = get(basis.lookup, encoded, 0)
            row == 0 && continue
            transition = conjugated ? conj(amplitude) : amplitude
            source = transposed ? row : column
            destination = transposed ? column : row
            input = vector[source]
            iszero(input) || (result[destination] += transition * input)
        end
    end
    return result
end

function _apply_discrete_terms!(
    result::AbstractVector,
    basis::Basis.DiscreteBasis,
    terms,
    vector::AbstractVector,
    alpha::Number=1,
    transposed::Bool=false,
    conjugated::Bool=false,
)
    if Basis._has_symmetry(basis.symmetry)
        parent = Basis._parent_basis_for_checks(basis)
        projector = basis.symmetry.projector
        ordinary_projection = transposed == conjugated
        lifted = ordinary_projection ?
            projector * vector :
            conj(projector) * vector
        acted = zeros(
            promote_type(eltype(lifted), _coefficient_type(terms)),
            length(parent),
        )
        _apply_discrete_terms!(
            acted,
            parent,
            terms,
            lifted,
            one(alpha),
            transposed,
            conjugated,
        )
        projection = ordinary_projection ? projector' : transpose(projector)
        mul!(result, projection, acted, alpha, true)
        return result
    end
    for term in terms
        _apply_discrete_term!(
            result,
            basis,
            term,
            vector,
            alpha,
            transposed,
            conjugated,
        )
    end
    return result
end

function _apply_discrete_terms(
    basis::Basis.DiscreteBasis,
    terms,
    vector::AbstractVector,
)
    T = promote_type(eltype(vector), _coefficient_type(terms))
    result = zeros(T, length(basis))
    return _apply_discrete_terms!(result, basis, terms, vector)
end

function _apply_user_callback_actions!(
    result::AbstractVector,
    basis::Basis.UserBasis,
    actions::Tuple,
    vector::AbstractVector,
    coefficient,
    transposed::Bool,
    conjugated::Bool,
)
    for (column, initial) in pairs(basis.base.encoded_states)
        encoded = initial
        amplitude = complex(coefficient)
        for (definition, site, _) in actions
            encoded, factor =
                Basis._user_callback_entry(definition, encoded, site)
            amplitude *= factor
        end
        row = get(basis.base.lookup, encoded, 0)
        row == 0 && continue
        transition = conjugated ? conj(amplitude) : amplitude
        source = transposed ? row : column
        destination = transposed ? column : row
        input = vector[source]
        iszero(input) || (result[destination] += transition * input)
    end
    return result
end

function _apply_user_term!(
    result::AbstractVector,
    basis::Basis.UserBasis,
    term::OperatorTerm,
    vector::AbstractVector,
    alpha::Number=1,
    transposed::Bool=false,
    conjugated::Bool=false,
)
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.N
    ]
    for coupling in term.couplings
        actions = Tuple(
            (
                Basis._user_operator(basis, op),
                Int(site),
                weights[Int(site)],
            )
            for (op, site) in Iterators.reverse(
                collect(zip(term.op, coupling[2:end])),
            )
        )
        coefficient = alpha * first(coupling)
        if all(action -> action[1] isa Function, actions)
            _apply_user_callback_actions!(
                result,
                basis,
                actions,
                vector,
                coefficient,
                transposed,
                conjugated,
            )
            continue
        end
        for (column, initial) in pairs(basis.base.encoded_states)
            branches = Dict(initial => complex(coefficient))
            for (definition, site, weight) in actions
                next_branches = Dict{UInt64,ComplexF64}()
                for (encoded, amplitude) in branches
                    if definition isa AbstractMatrix
                        old = Int(
                            (encoded ÷ weight) %
                            UInt64(basis.sps),
                        )
                        for new in 0:(basis.sps - 1)
                            factor = definition[new + 1, old + 1]
                            iszero(factor) && continue
                            updated = UInt64(
                                Int128(encoded) +
                                Int128(new - old) * Int128(weight),
                            )
                            next_branches[updated] =
                                get(next_branches, updated, 0) + amplitude * factor
                        end
                    else
                        updated, factor = definition(encoded, site)
                        next_branches[UInt64(updated)] =
                            get(next_branches, UInt64(updated), 0) + amplitude * factor
                    end
                end
                branches = next_branches
            end
            for (encoded, amplitude) in branches
                row = get(basis.base.lookup, encoded, 0)
                row == 0 && continue
                transition = conjugated ? conj(amplitude) : amplitude
                source = transposed ? row : column
                destination = transposed ? column : row
                input = vector[source]
                iszero(input) || (result[destination] += transition * input)
            end
        end
    end
    return result
end

function _apply_user_terms!(
    result::AbstractVector,
    basis::Basis.UserBasis,
    terms,
    vector::AbstractVector,
    alpha::Number=1,
    transposed::Bool=false,
    conjugated::Bool=false,
)
    for term in terms
        _apply_user_term!(
            result,
            basis,
            term,
            vector,
            alpha,
            transposed,
            conjugated,
        )
    end
    return result
end

function _apply_user_terms(
    basis::Basis.UserBasis,
    terms,
    vector::AbstractVector,
)
    T = promote_type(eltype(vector), _coefficient_type(terms))
    result = zeros(T, length(basis))
    return _apply_user_terms!(result, basis, terms, vector)
end

_apply_terms!(
    result,
    basis::SpinBasis1D,
    terms,
    vector,
    alpha=1,
    transposed=false,
    conjugated=false,
) = _apply_spin_terms!(
    result,
    basis,
    terms,
    vector,
    alpha,
    transposed,
    conjugated,
)
_apply_terms!(
    result,
    basis::Basis.DiscreteBasis,
    terms,
    vector,
    alpha=1,
    transposed=false,
    conjugated=false,
) = _apply_discrete_terms!(
    result,
    basis,
    terms,
    vector,
    alpha,
    transposed,
    conjugated,
)
_apply_terms!(
    result,
    basis::Basis.UserBasis,
    terms,
    vector,
    alpha=1,
    transposed=false,
    conjugated=false,
) = _apply_user_terms!(
    result,
    basis,
    terms,
    vector,
    alpha,
    transposed,
    conjugated,
)
function _apply_terms!(
    result,
    basis::AbstractBasis,
    terms,
    vector,
    alpha=1,
    transposed=false,
    conjugated=false,
)
    for term in terms
        matrix = Basis.operator_matrix(
            basis,
            term.op,
            term.couplings;
            sparse=true,
        )
        transposed && (matrix = transpose(matrix))
        conjugated && (matrix = conj(matrix))
        mul!(result, matrix, vector, alpha, true)
    end
    return result
end

_apply_terms(basis::SpinBasis1D, terms, vector) =
    _apply_spin_terms(basis, terms, vector)
_apply_terms(basis::Basis.DiscreteBasis, terms, vector) =
    _apply_discrete_terms(basis, terms, vector)
_apply_terms(basis::Basis.UserBasis, terms, vector) =
    _apply_user_terms(basis, terms, vector)
_apply_terms(basis::AbstractBasis, terms, vector) =
    sum(
        Basis.operator_matrix(basis, term.op, term.couplings) * vector
        for term in terms
    )

function _structured_is_hermitian(terms)
    coefficients = Dict{Tuple{String,Tuple},Any}()
    for term in terms, coupling in term.couplings
        key = (term.op, Tuple(coupling[2:end]))
        coefficients[key] =
            get(coefficients, key, zero(first(coupling))) + first(coupling)
    end
    isempty(coefficients) && return true
    adjoint_character = Dict(
        'I' => 'I',
        'x' => 'x',
        'y' => 'y',
        'z' => 'z',
        'n' => 'n',
        '+' => '-',
        '-' => '+',
        '|' => '|',
    )
    for ((op, sites), coefficient) in coefficients
        all(character -> haskey(adjoint_character, character), op) ||
            return nothing
        adjoint_op = String([adjoint_character[character] for character in op])
        partner = get(coefficients, (adjoint_op, sites), nothing)
        partner === nothing && return false
        isapprox(partner, conj(coefficient); atol=3e-12, rtol=3e-12) ||
            return false
    end
    return true
end

function _matrixfree_is_hermitian(basis, terms)
    dimension = length(basis)
    dimension == 0 && return true
    for seed in 1:min(3, dimension)
        left = ComplexF64[
            cis((seed + 1) * index) / sqrt(dimension)
            for index in 1:dimension
        ]
        right = ComplexF64[
            cis((seed + 2) * index^2 / max(1, dimension)) / sqrt(dimension)
            for index in 1:dimension
        ]
        left_action = _apply_terms(basis, terms, left)
        right_action = _apply_terms(basis, terms, right)
        isapprox(
            dot(left, right_action),
            dot(left_action, right);
            atol=3e-11,
            rtol=3e-11,
        ) || return false
    end
    return true
end

function _apply_linear_base(
    operator::QuantumLinearOperator,
    value::AbstractVector,
)
    T = promote_type(eltype(operator), eltype(value))
    result = zeros(T, size(operator, 1))
    if operator.explicit_data === nothing
        _apply_terms!(
            result,
            operator.basis,
            operator.action_terms,
            value,
            1,
            operator.transposed,
            operator.conjugated,
        )
    else
        matrix = operator.transposed ?
            transpose(operator.explicit_data) :
            operator.explicit_data
        operator.conjugated && (matrix = conj(matrix))
        mul!(result, matrix, value)
    end
    return result
end

function _apply_linear_base(
    operator::QuantumLinearOperator,
    value::AbstractMatrix,
)
    T = promote_type(eltype(operator), eltype(value))
    result = zeros(T, size(operator, 1), size(value, 2))
    for column in axes(value, 2)
        output = @view result[:, column]
        input = @view value[:, column]
        if operator.explicit_data === nothing
            _apply_terms!(
                output,
                operator.basis,
                operator.action_terms,
                input,
                1,
                operator.transposed,
                operator.conjugated,
            )
        else
            matrix = operator.transposed ?
                transpose(operator.explicit_data) :
                operator.explicit_data
            operator.conjugated && (matrix = conj(matrix))
            mul!(output, matrix, input)
        end
    end
    return result
end

function _apply_linear(operator::QuantumLinearOperator, value::AbstractVecOrMat)
    size(value, 1) == length(operator.diagonal) ||
        throw(DimensionMismatch("operator and value dimensions do not match"))
    result = _apply_linear_base(operator, value)
    _add_qlo_diagonal!(result, value, operator, true)
    return result
end

function _add_qlo_diagonal!(output, input, operator, alpha)
    if input isa AbstractVector
        @inbounds for index in eachindex(input)
            coefficient =
                operator.diagonal[index] +
                operator.extracted_diagonal[index]
            operator.conjugated && (coefficient = conj(coefficient))
            output[index] += alpha * coefficient * input[index]
        end
    else
        @inbounds for column in axes(input, 2), row in axes(input, 1)
            coefficient =
                operator.diagonal[row] +
                operator.extracted_diagonal[row]
            operator.conjugated && (coefficient = conj(coefficient))
            output[row, column] +=
                alpha * coefficient * input[row, column]
        end
    end
    return output
end

function _linear_data(operator::QuantumLinearOperator)
    dimension = size(operator, 1)
    return _apply_linear(
        operator,
        Matrix{eltype(operator)}(I, dimension, dimension),
    )
end

Base.Matrix(operator::QuantumLinearOperator) = Matrix(_linear_data(operator))
Base.size(operator::QuantumLinearOperator) =
    (length(operator.diagonal), length(operator.diagonal))
Base.size(operator::QuantumLinearOperator, dimension::Integer) =
    size(operator)[dimension]
Base.eltype(::Type{<:QuantumLinearOperator{T}}) where {T} = T
Base.eltype(operator::QuantumLinearOperator) = eltype(typeof(operator))
Base.:*(
    operator::QuantumLinearOperator{T},
    value::AbstractVector{S},
) where {T,S} = _apply_linear(operator, value)
Base.:*(
    operator::QuantumLinearOperator{T},
    value::AbstractMatrix{S},
) where {T,S} = _apply_linear(operator, value)
LinearAlgebra.mul!(
    output::AbstractVector,
    operator::QuantumLinearOperator,
    value::AbstractVector,
) = mul!(output, operator, value, true, false)
function LinearAlgebra.mul!(
    output::AbstractVector,
    operator::QuantumLinearOperator,
    value::AbstractVector,
    alpha::Number,
    beta::Number,
)
    length(output) == size(operator, 1) ||
        throw(DimensionMismatch("output and operator dimensions do not match"))
    length(value) == size(operator, 2) ||
        throw(DimensionMismatch("operator and value dimensions do not match"))
    input = Base.mightalias(output, value) ? copy(value) : value
    iszero(beta) ? fill!(output, zero(eltype(output))) : lmul!(beta, output)
    if operator.explicit_data === nothing
        _apply_terms!(
            output,
            operator.basis,
            operator.action_terms,
            input,
            alpha,
            operator.transposed,
            operator.conjugated,
        )
    else
        matrix = operator.transposed ?
            transpose(operator.explicit_data) :
            operator.explicit_data
        operator.conjugated && (matrix = conj(matrix))
        mul!(output, matrix, input, alpha, true)
    end
    _add_qlo_diagonal!(output, input, operator, alpha)
    return output
end
LinearAlgebra.mul!(
    output::AbstractMatrix,
    operator::QuantumLinearOperator,
    value::AbstractMatrix,
) = mul!(output, operator, value, true, false)
function LinearAlgebra.mul!(
    output::AbstractMatrix,
    operator::QuantumLinearOperator,
    value::AbstractMatrix,
    alpha::Number,
    beta::Number,
)
    size(output) == (size(operator, 1), size(value, 2)) ||
        throw(DimensionMismatch("output has the wrong dimensions"))
    size(value, 1) == size(operator, 2) ||
        throw(DimensionMismatch("operator and value dimensions do not match"))
    input = Base.mightalias(output, value) ? copy(value) : value
    iszero(beta) ? fill!(output, zero(eltype(output))) : lmul!(beta, output)
    for column in axes(input, 2)
        mul!(
            @view(output[:, column]),
            operator,
            @view(input[:, column]),
            alpha,
            true,
        )
    end
    return output
end
LinearAlgebra.ishermitian(operator::QuantumLinearOperator) = operator.hermitian
LinearAlgebra.issymmetric(operator::QuantumLinearOperator) =
    eltype(operator) <: Real && operator.hermitian

function Base.getproperty(operator::QuantumLinearOperator, name::Symbol)
    name === :H && return adjoint(operator)
    name === :T && return transpose(operator)
    name === :Ns && return length(getfield(operator, :diagonal))
    name === :dtype && return eltype(typeof(operator))
    name === :get_shape && return size(operator)
    name === :ndim && return 2
    name === :shape && return size(operator)
    return getfield(operator, name)
end

function _linear_from_matrix(
    source::QuantumLinearOperator,
    matrix::AbstractMatrix,
)
    T = eltype(matrix)
    static_list = copy(source.static_list)
    action_terms = copy(source.action_terms)
    return QuantumLinearOperator{
        T,
        typeof(source.basis),
        typeof(static_list),
        typeof(action_terms),
    }(
        source.basis,
        static_list,
        action_terms,
        Matrix{T}(matrix),
        zeros(T, size(matrix, 1)),
        zeros(T, size(matrix, 1)),
        ishermitian(matrix),
        false,
        false,
    )
end

Base.copy(operator::QuantumLinearOperator) = deepcopy(operator)
function _linear_with_flags(operator::QuantumLinearOperator, toggle_t, toggle_c)
    result = copy(operator)
    result.transposed = xor(result.transposed, toggle_t)
    result.conjugated = xor(result.conjugated, toggle_c)
    return result
end
Base.transpose(operator::QuantumLinearOperator) =
    _linear_with_flags(operator, true, false)
Base.conj(operator::QuantumLinearOperator) =
    _linear_with_flags(operator, false, true)
Base.adjoint(operator::QuantumLinearOperator) =
    _linear_with_flags(operator, true, true)
isquantum_LinearOperator(value) = value isa QuantumLinearOperator

function set_diagonal!(
    operator::QuantumLinearOperator,
    diagonal;
    copy::Bool=true,
)
    length(diagonal) == size(operator, 1) ||
        throw(DimensionMismatch("diagonal must have length Ns"))
    operator.diagonal .= diagonal
    return operator
end

function apply(
    operator::QuantumLinearOperator,
    value;
    out=nothing,
    a::Number=1,
    kwargs...,
)
    T = promote_type(eltype(operator), eltype(value), typeof(a))
    result = out === nothing ?
        _action_output(T, size(operator, 1), value) :
        out
    mul!(result, operator, value, a, false)
    return result
end

function right_apply(
    operator::QuantumLinearOperator,
    value;
    out=nothing,
    a::Number=1,
    kwargs...,
)
    if value isa StridedVector{<:LinearAlgebra.BlasFloat} &&
       size(operator, 1) <= 64
        T = promote_type(eltype(operator), eltype(value), typeof(a))
        if T <: LinearAlgebra.BlasFloat
            result = out === nothing ? nothing : out
            result === nothing ||
                length(result) == size(operator, 2) ||
                throw(DimensionMismatch("out must have length Ns"))
            input = result !== nothing && Base.mightalias(result, value) ?
                copy(value) :
                value
            # Match the platform BLAS reduction order used by
            # transpose(value) * Matrix(operator).  This exact-compatibility
            # crossover is deliberately restricted to tiny operators; the
            # general path below remains matrix-free.
            product = vec(transpose(input) * Matrix(operator))
            isone(a) || lmul!(a, product)
            result === nothing && return product
            copyto!(result, product)
            return result
        end
    end
    transposed_operator = transpose(operator)
    transposed_value =
        value isa AbstractVector ? value : transpose(value)
    if value isa AbstractVector
        return apply(transposed_operator, value; out, a)
    end
    T = promote_type(eltype(operator), eltype(value), typeof(a))
    result = out === nothing ?
        zeros(T, size(value, 1), size(operator, 2)) :
        out
    mul!(transpose(result), transposed_operator, transposed_value, a, false)
    return result
end

function eigsh(
    operator::QuantumLinearOperator;
    k::Integer=min(6, size(operator, 1)),
    which=:SA,
    return_eigenvectors::Bool=true,
    kwargs...,
)
    n = size(operator, 1)
    1 <= k <= n || throw(ArgumentError("k must lie in 1:Ns"))
    requested = Symbol(uppercase(String(which)))
    requested in (:SA, :LA, :SM, :LM, :BE) ||
        throw(ArgumentError("which must be SA, LA, SM, LM, or BE"))
    if k >= n - 1
        H = _hamiltonian_from_data(operator.basis, _linear_data(operator))
        return eigsh(H; k, which, return_eigenvectors, kwargs...)
    end
    arpack_which = requested === :SA ? :SR : requested === :LA ? :LR : requested
    arpack_result = Arpack.eigs(
        operator;
        nev=k,
        which=arpack_which,
        ritzvec=return_eigenvectors,
        kwargs...,
    )
    values = arpack_result[1]
    vectors = return_eigenvectors ? arpack_result[2] : nothing
    order = requested === :SA ?
        sortperm(real.(values)) :
        requested === :LA ?
        sortperm(real.(values); rev=true) :
        requested === :SM ?
        sortperm(abs.(values)) :
        requested === :LM ?
        sortperm(abs.(values); rev=true) :
        eachindex(values)
    selected_values = operator.hermitian ?
        real.(values[order]) :
        values[order]
    return return_eigenvectors ?
        (selected_values, vectors[:, order]) :
        selected_values
end

function expt_value(
    operator::QuantumLinearOperator,
    state;
    enforce_pure::Bool=false,
    kwargs...,
)
    if state isa AbstractVector
        return dot(state, operator * state)
    elseif enforce_pure || size(state, 2) != size(operator, 1)
        return [dot(column, operator * column) for column in eachcol(state)]
    end
    return tr(right_apply(operator, state))
end

function matrix_ele(
    operator::QuantumLinearOperator,
    left,
    right;
    diagonal::Bool=false,
    kwargs...,
)
    elements = left' * (operator * right)
    return diagonal && elements isa AbstractMatrix ? diag(elements) : elements
end

function quant_fluct(
    operator::QuantumLinearOperator,
    state;
    enforce_pure::Bool=false,
    kwargs...,
)
    mean = expt_value(operator, state; enforce_pure, kwargs...)
    if state isa AbstractVector
        second = dot(state, operator * (operator * state))
        return second - mean^2
    end
    if ndims(state) == 3
        return [
            quant_fluct(
                operator,
                @view(state[:, :, index]);
                enforce_pure,
                kwargs...,
            )
            for index in axes(state, 3)
        ]
    end
    first_action = operator * state
    second = tr(right_apply(operator, first_action))
    return second - mean^2
end

"""
    QuantumOperator(basis, input_dict)

Parameter-dependent operator `sum(pars[key] * input_dict[key])`. Dictionary
values may be native `OperatorTerm` vectors or square matrices.
"""
mutable struct QuantumOperator{T<:Number}
    basis::AbstractBasis
    components::Dict{Any,NativeMatrix{T}}
end

"""
    save_zip(archive, operator; save_basis=true)

Persist a `QuantumOperator` in QuSpin.jl's versioned native archive format.
The function retains the historical name while deliberately avoiding a Python
pickle dependency.
"""
function save_zip(
    archive::AbstractString,
    operator::QuantumOperator;
    save_basis::Bool=true,
)
    payload = Dict(
        "format" => "QuSpin.jl-quantum-operator-v1",
        "basis" => save_basis ? operator.basis : nothing,
        "components" => operator.components,
    )
    open(archive, "w") do io
        serialize(io, payload)
    end
    return archive
end

save_zip(archive::AbstractString, operator::QuantumOperator, save_basis::Bool) =
    save_zip(archive, operator; save_basis)

"""
    load_zip(archive)

Load an archive written by `save_zip`.
"""
function load_zip(archive::AbstractString)
    payload = open(deserialize, archive)
    payload isa AbstractDict &&
        get(payload, "format", nothing) == "QuSpin.jl-quantum-operator-v1" ||
        throw(ArgumentError("unsupported QuSpin operator archive"))
    components = payload["components"]
    basis = payload["basis"]
    if basis === nothing
        dimension = size(first(values(components)), 1)
        ispow2(dimension) ||
            throw(ArgumentError("an archive without a basis must have a power-of-two dimension"))
        basis = SpinBasis1D(trailing_zeros(dimension))
    end
    return QuantumOperator(basis, components)
end

function QuantumOperator(
    basis::AbstractBasis,
    input_dict::AbstractDict,
    ;
    matrix_formats::AbstractDict=Dict(),
)
    isempty(input_dict) && throw(ArgumentError("input_dict must be nonempty"))
    raw = Dict{Any,Any}()
    raw_formats = Dict{Any,Symbol}()
    types = Type[]
    for (key, value) in input_dict
        requested_format = haskey(matrix_formats, key) ?
            matrix_formats[key] :
            value isa SparseMatrixCSC ? :csc : :dense
        matrix = if value isa AbstractMatrix
            value
        elseif value isa AbstractVector{<:OperatorTerm}
            Hamiltonian(
                basis,
                value;
                static_fmt=requested_format,
            ).data
        else
            throw(ArgumentError("operator components must be matrices or OperatorTerm vectors"))
        end
        size(matrix) == (length(basis), length(basis)) ||
            throw(DimensionMismatch("every component must have shape (Ns,Ns)"))
        raw[key] = matrix
        raw_formats[key] = _normalize_matrix_format(requested_format)
        push!(types, eltype(matrix))
    end
    T = promote_type(types...)
    components = Dict{Any,NativeMatrix{T}}(
        key => _matrix_with_format(
            value,
            T,
            raw_formats[key],
        )
        for (key, value) in raw
    )
    return QuantumOperator{T}(basis, components)
end

function _parameter_matrix(operator::QuantumOperator, pars::AbstractDict=Dict())
    T = promote_type(
        eltype(first(values(operator.components))),
        (typeof(value) for value in values(pars))...,
    )
    result = all(value isa SparseMatrixCSC for value in values(operator.components)) ?
        spzeros(T, size(operator)...) :
        zeros(T, size(operator)...)
    for (key, matrix) in operator.components
        result = result + get(pars, key, zero(T)) * matrix
    end
    return result
end

Base.size(operator::QuantumOperator) = size(first(values(operator.components)))
Base.size(operator::QuantumOperator, dimension::Integer) =
    size(first(values(operator.components)), dimension)
Base.eltype(operator::QuantumOperator) = eltype(first(values(operator.components)))

function Base.getproperty(operator::QuantumOperator, name::Symbol)
    name === :H && return adjoint(operator)
    name === :T && return transpose(operator)
    name === :Ns && return size(first(values(getfield(operator, :components))), 1)
    name === :dtype && return eltype(first(values(getfield(operator, :components))))
    name === :get_shape && return size(first(values(getfield(operator, :components))))
    name === :is_dense &&
        return any(value isa Matrix for value in values(getfield(operator, :components)))
    name === :ndim && return 2
    name === :shape && return size(first(values(getfield(operator, :components))))
    return getfield(operator, name)
end

function _quantum_operator_from_components(source::QuantumOperator, components)
    T = promote_type((eltype(value) for value in values(components))...)
    return QuantumOperator{T}(
        source.basis,
        Dict{Any,NativeMatrix{T}}(
            key => _matrix_with_format(
                value,
                T,
                _storage_format(source.components[key]),
            )
            for (key, value) in components
        ),
    )
end

Base.copy(operator::QuantumOperator) = deepcopy(operator)
Base.transpose(operator::QuantumOperator) = _quantum_operator_from_components(
    operator,
    Dict(key => transpose(value) for (key, value) in operator.components),
)
Base.conj(operator::QuantumOperator) = _quantum_operator_from_components(
    operator,
    Dict(key => conj(value) for (key, value) in operator.components),
)
Base.adjoint(operator::QuantumOperator) = _quantum_operator_from_components(
    operator,
    Dict(key => adjoint(value) for (key, value) in operator.components),
)
isquantum_operator(value) = value isa QuantumOperator

get_operators(operator::QuantumOperator, key) = copy(operator.components[key])

function astype(
    operator::QuantumOperator,
    ::Type{T};
    copy::Bool=false,
    kwargs...,
) where {T<:Number}
    return QuantumOperator{T}(
        operator.basis,
        Dict{Any,NativeMatrix{T}}(
            key => _matrix_with_format(value, T, _storage_format(value))
            for (key, value) in operator.components
        ),
    )
end

toarray(operator::QuantumOperator; pars::AbstractDict=Dict(), out=nothing) =
    _copy_or_write(Matrix(_parameter_matrix(operator, pars)), out)
todense(operator::QuantumOperator; kwargs...) = toarray(operator; kwargs...)
tocsc(operator::QuantumOperator; pars::AbstractDict=Dict()) =
    sparse(_parameter_matrix(operator, pars))
function tocsr(operator::QuantumOperator; pars::AbstractDict=Dict())
    return SparseMatrixCSR(_parameter_matrix(operator, pars))
end
function diagonal(operator::QuantumOperator; pars::AbstractDict=Dict())
    T = promote_type(
        eltype(operator),
        (typeof(value) for value in values(pars))...,
    )
    result = zeros(T, size(operator, 1))
    for (key, matrix) in operator.components
        coefficient = get(pars, key, zero(T))
        iszero(coefficient) || (result .+= coefficient .* diag(matrix))
    end
    return result
end
function LinearAlgebra.tr(
    operator::QuantumOperator;
    pars::AbstractDict=Dict(),
)
    T = promote_type(
        eltype(operator),
        (typeof(value) for value in values(pars))...,
    )
    result = zero(T)
    for (key, matrix) in operator.components
        coefficient = get(pars, key, zero(T))
        iszero(coefficient) || (result += coefficient * tr(matrix))
    end
    return result
end

function apply(
    operator::QuantumOperator,
    value::AbstractVecOrMat;
    pars::AbstractDict=Dict(),
    out=nothing,
    overwrite_out::Bool=true,
    a::Number=1,
    kwargs...,
)
    size(value, 1) == size(operator, 2) ||
        throw(DimensionMismatch("operator and value dimensions do not match"))
    T = promote_type(
        eltype(operator),
        eltype(value),
        typeof(a),
        (typeof(coefficient) for coefficient in values(pars))...,
    )
    result = out === nothing ?
        _action_output(T, size(operator, 1), value) :
        out
    expected_axes = value isa AbstractVector ?
        (axes(first(values(operator.components)), 1),) :
        (
            axes(first(values(operator.components)), 1),
            axes(value, 2),
        )
    axes(result) == expected_axes ||
        throw(DimensionMismatch("out must have the same axes as the result"))
    input = Base.mightalias(result, value) ? copy(value) : value
    if out === nothing || overwrite_out
        fill!(result, zero(eltype(result)))
    end
    for (key, matrix) in operator.components
        coefficient = get(pars, key, zero(T))
        iszero(coefficient) && continue
        _matrix_mul_add!(result, matrix, input, a * coefficient, true)
    end
    return result
end

function right_apply(
    operator::QuantumOperator,
    value;
    pars::AbstractDict=Dict(),
    out=nothing,
    overwrite_out::Bool=true,
    a::Number=1,
    kwargs...,
)
    T = promote_type(
        eltype(operator),
        eltype(value),
        typeof(a),
        (typeof(coefficient) for coefficient in values(pars))...,
    )
    result = out === nothing ? similar(value, T) : out
    if value isa StridedVector{<:LinearAlgebra.BlasFloat} &&
       T <: LinearAlgebra.BlasFloat &&
       operator.is_dense &&
       size(operator, 1) <= 64
        length(result) == size(operator, 2) ||
            throw(DimensionMismatch("out must have length Ns"))
        input = Base.mightalias(result, value) ? copy(value) : value
        column = zeros(T, size(operator, 1))
        @inbounds for destination in eachindex(result)
            fill!(column, zero(T))
            for (key, matrix) in operator.components
                coefficient = get(pars, key, zero(T))
                iszero(coefficient) && continue
                for source in eachindex(column)
                    column[source] +=
                        coefficient * matrix[source, destination]
                end
            end
            product = a * LinearAlgebra.BLAS.dotu(
                length(input),
                input,
                stride(input, 1),
                column,
                1,
            )
            if out === nothing || overwrite_out
                result[destination] = product
            else
                result[destination] += product
            end
        end
        return result
    end
    out === nothing || overwrite_out ?
        fill!(result, zero(eltype(result))) :
        nothing
    input = Base.mightalias(result, value) ? copy(value) : value
    for (key, matrix) in operator.components
        coefficient = get(pars, key, zero(T))
        iszero(coefficient) && continue
        _right_matrix_mul_add!(
            result,
            input,
            matrix,
            a * coefficient,
            true,
        )
    end
    return result
end

function eigh(operator::QuantumOperator; pars::AbstractDict=Dict(), kwargs...)
    matrix = Matrix(_parameter_matrix(operator, pars))
    decomposition = ishermitian(matrix) ?
        eigen(Hermitian(matrix)) :
        eigen(matrix)
    return decomposition.values, decomposition.vectors
end

function LinearAlgebra.eigvals(
    operator::QuantumOperator;
    pars::AbstractDict=Dict(),
)
    values, _ = eigh(operator; pars)
    return values
end

function eigsh(
    operator::QuantumOperator;
    pars::AbstractDict=Dict(),
    k::Integer=min(6, size(operator, 1)),
    which=:SA,
    kwargs...,
)
    H = _hamiltonian_from_data(
        operator.basis,
        _parameter_matrix(operator, pars),
    )
    return eigsh(H; k, which, kwargs...)
end

function tohamiltonian(
    operator::QuantumOperator;
    pars::AbstractDict=Dict(),
    copy::Bool=true,
)
    return _hamiltonian_from_data(
        operator.basis,
        _parameter_matrix(operator, pars),
    )
end

function aslinearoperator(
    operator::QuantumOperator;
    pars::AbstractDict=Dict(),
)
    T = promote_type(
        eltype(operator),
        (typeof(value) for value in values(pars))...,
    )
    action! = (output, input) ->
        apply(
            operator,
            input;
            pars,
            out=output,
            overwrite_out=true,
        )
    hermitian = all(
        iszero(get(pars, key, zero(T))) ||
        (
            isreal(get(pars, key, zero(T))) &&
            ishermitian(matrix)
        )
        for (key, matrix) in operator.components
    )
    return ActionLinearOperator{T,typeof(action!)}(
        size(operator, 1),
        action!,
        hermitian,
    )
end

function expt_value(
    operator::QuantumOperator,
    state;
    pars::AbstractDict=Dict(),
    enforce_pure::Bool=false,
    kwargs...,
)
    if state isa AbstractVector
        return dot(state, apply(operator, state; pars))
    elseif enforce_pure || (ndims(state) == 2 && size(state, 2) != size(operator, 1))
        acted = apply(operator, state; pars)
        return [
            dot(@view(state[:, column]), @view(acted[:, column]))
            for column in axes(state, 2)
        ]
    end
    if ndims(state) == 3
        return [
            expt_value(
                operator,
                @view(state[:, :, index]);
                pars,
            )
            for index in axes(state, 3)
        ]
    end
    T = promote_type(
        eltype(operator),
        eltype(state),
        (typeof(value) for value in values(pars))...,
    )
    result = zero(T)
    for (key, matrix) in operator.components
        coefficient = get(pars, key, zero(T))
        iszero(coefficient) ||
            (result += coefficient * _trace_product(state, matrix))
    end
    return result
end

function matrix_ele(
    operator::QuantumOperator,
    left,
    right;
    pars::AbstractDict=Dict(),
    diagonal::Bool=false,
    kwargs...,
)
    elements = left' * apply(operator, right; pars)
    return diagonal && elements isa AbstractMatrix ? diag(elements) : elements
end

function quant_fluct(
    operator::QuantumOperator,
    state;
    pars::AbstractDict=Dict(),
    enforce_pure::Bool=false,
    kwargs...,
)
    if state isa AbstractVector
        mean = expt_value(operator, state; pars)
        first_action = apply(operator, state; pars)
        second = dot(state, apply(operator, first_action; pars))
        return second - mean^2
    elseif enforce_pure || (ndims(state) == 2 && size(state, 2) != size(operator, 1))
        mean = expt_value(operator, state; pars, enforce_pure=true)
        first_action = apply(operator, state; pars)
        second_action = apply(operator, first_action; pars)
        second = [
            dot(@view(state[:, column]), @view(second_action[:, column]))
            for column in axes(state, 2)
        ]
        return second .- mean .^ 2
    end
    mean = expt_value(operator, state; pars)
    first_action = apply(operator, state; pars)
    second = tr(right_apply(operator, first_action; pars))
    return second - mean^2
end

function update_matrix_formats!(
    operator::QuantumOperator,
    matrix_formats::AbstractDict,
)
    for (key, requested) in matrix_formats
        haskey(operator.components, key) ||
            throw(KeyError(key))
        operator.components[key] = _matrix_with_format(
            operator.components[key],
            eltype(operator),
            requested,
            copy_data=false,
        )
    end
    return operator
end

_operator_matrix(operator::Hamiltonian) = operator.data
_operator_matrix(operator::AbstractMatrix) = operator

"""Return the matrix commutator `A * B - B * A`."""
function commutator(left, right)
    A = _operator_matrix(left)
    B = _operator_matrix(right)
    size(A, 2) == size(B, 1) && size(B, 2) == size(A, 1) ||
        throw(DimensionMismatch("operators must have mutually compatible dimensions"))
    return A * B - B * A
end

"""Return the matrix anticommutator `A * B + B * A`."""
function anti_commutator(left, right)
    A = _operator_matrix(left)
    B = _operator_matrix(right)
    size(A, 2) == size(B, 1) && size(B, 2) == size(A, 1) ||
        throw(DimensionMismatch("operators must have mutually compatible dimensions"))
    return A * B + B * A
end

function _operator_dense(operator)
    return Matrix(operator)
end

"""
    ExpOp(O; a=1, start=nothing, stop=nothing, num=nothing,
          endpoint=true, iterate=false)

Julia-native matrix-exponential action. A configured grid evaluates
`exp(a * grid[i] * O)` without requiring Python or SciPy.
"""
mutable struct ExpOp{M}
    O::M
    a::Number
    grid::Union{Nothing,Vector{Float64}}
    step::Union{Nothing,Float64}
    iterate::Bool
end

function _grid(start, stop, num, endpoint)
    start isa Real && stop isa Real ||
        throw(ArgumentError("grid endpoints must be real scalars"))
    count = num === nothing ? 50 : Int(num)
    count >= 1 || throw(ArgumentError("num must be positive"))
    include_endpoint = endpoint === nothing ? true : Bool(endpoint)
    if count == 1
        return [Float64(start)], NaN
    end
    step = include_endpoint ?
        (Float64(stop) - Float64(start)) / (count - 1) :
        (Float64(stop) - Float64(start)) / count
    return [Float64(start) + (index - 1) * step for index in 1:count], step
end

function ExpOp(
    O;
    a::Number=1.0,
    start=nothing,
    stop=nothing,
    num=nothing,
    endpoint=nothing,
    iterate::Bool=false,
)
    size(O, 1) == size(O, 2) ||
        throw(ArgumentError("O must be square"))
    if start === nothing && stop === nothing
        num === nothing && endpoint === nothing ||
            throw(ArgumentError("num and endpoint require a grid"))
        iterate && throw(ArgumentError("iterate=true requires a grid"))
        grid, step = nothing, nothing
    elseif start === nothing || stop === nothing
        throw(ArgumentError("both start and stop are required"))
    else
        grid, step = _grid(start, stop, num, endpoint)
    end
    return ExpOp(O, a, grid, step, iterate)
end

function Base.getproperty(operator::ExpOp, name::Symbol)
    name === :H && return adjoint(operator)
    name === :T && return transpose(operator)
    name === :Ns && return size(getfield(operator, :O), 1)
    name === :get_shape && return size(getfield(operator, :O))
    name === :ndim && return 2
    return getfield(operator, name)
end

Base.size(operator::ExpOp) = operator.get_shape
Base.copy(operator::ExpOp) = deepcopy(operator)
isexp_op(value) = value isa ExpOp

function Base.transpose(operator::ExpOp)
    return _copy_exp_with_operator(operator, transpose(operator.O), operator.a)
end

function Base.conj(operator::ExpOp)
    return _copy_exp_with_operator(operator, conj(operator.O), conj(operator.a))
end

function Base.adjoint(operator::ExpOp)
    return _copy_exp_with_operator(operator, adjoint(operator.O), conj(operator.a))
end

function _copy_exp_with_operator(source::ExpOp, O, a)
    result = ExpOp(O; a)
    result.grid = source.grid === nothing ? nothing : copy(source.grid)
    result.step = source.step
    result.iterate = source.iterate
    return result
end

function set_a!(operator::ExpOp, value::Number)
    operator.a = value
    return operator
end

function set_grid!(
    operator::ExpOp,
    start::Real,
    stop::Real;
    num::Integer=50,
    endpoint::Bool=true,
)
    operator.grid, operator.step = _grid(start, stop, num, endpoint)
    return operator
end

function unset_grid!(operator::ExpOp)
    operator.grid = nothing
    operator.step = nothing
    operator.iterate = false
    return operator
end

function set_iterate!(operator::ExpOp, value::Bool)
    value && operator.grid === nothing &&
        throw(ArgumentError("iterate=true requires a grid"))
    operator.iterate = value
    return operator
end

function _exp_matrix(operator::ExpOp, scale::Number=1; shift=nothing)
    matrix = _operator_dense(operator.O)
    result = exp((operator.a * scale) * matrix)
    shift === nothing ||
        lmul!(exp(operator.a * scale * shift), result)
    return result
end

get_mat(operator::ExpOp; dense::Bool=true, shift=nothing, kwargs...) =
    dense ?
    _exp_matrix(operator; shift) :
    sparse(_exp_matrix(operator; shift))

_exp_source(operator::AbstractMatrix; kwargs...) = operator
_exp_source(operator::Hamiltonian; time=0, kwargs...) =
    aslinearoperator(operator; time)
_exp_source(operator::QuantumOperator; pars::AbstractDict=Dict(), kwargs...) =
    aslinearoperator(operator; pars)

function _exp_apply_single(
    operator::ExpOp,
    value::AbstractVecOrMat,
    scale::Number;
    shift=nothing,
    kwargs...,
)
    source = _exp_source(operator.O; kwargs...)
    result = _krylov_expmv(
        source,
        value,
        operator.a * scale;
        tol=get(kwargs, :tol, 1e-12),
        krylov_dim=get(kwargs, :krylov_dim, 30),
    )
    shift === nothing ||
        lmul!(exp(operator.a * scale * shift), result)
    return result
end

function _exp_grid_action(operator::ExpOp, value; shift=nothing, kwargs...)
    if operator.grid === nothing
        return _exp_apply_single(operator, value, 1; shift, kwargs...)
    end
    operator.iterate && return (
        _exp_apply_single(operator, value, scale; shift, kwargs...)
        for scale in operator.grid
    )
    if value isa AbstractVector
        source = _exp_source(operator.O; kwargs...)
        scales = operator.a .* operator.grid
        results = _krylov_expmv_times(
            source,
            value,
            scales;
            tol=get(kwargs, :tol, 1e-12),
            krylov_dim=get(kwargs, :krylov_dim, 30),
        )
        if shift !== nothing
            for (column, scale) in pairs(operator.grid)
                rmul!(
                    @view(results[:, column]),
                    exp(operator.a * scale * shift),
                )
            end
        end
        return results
    end
    source = _exp_source(operator.O; kwargs...)
    T = promote_type(
        eltype(source),
        eltype(value),
        typeof(operator.a),
        shift === nothing ? Float64 : typeof(shift),
    )
    results = Array{T}(
        undef,
        size(value, 1),
        size(value, 2),
        length(operator.grid),
    )
    for (index, scale) in pairs(operator.grid)
        state = _exp_apply_single(
            operator,
            value,
            scale;
            shift,
            kwargs...,
        )
        copyto!(@view(results[:, :, index]), state)
    end
    return results
end

function apply(operator::ExpOp, other; shift=nothing, kwargs...)
    value = _operator_dense_or_self(other)
    value isa AbstractVecOrMat ||
        throw(ArgumentError("exponential action requires a vector or matrix"))
    return _exp_grid_action(operator, value; shift, kwargs...)
end

_operator_dense_or_self(value::Hamiltonian) = Matrix(value)
_operator_dense_or_self(value) = value

function right_apply(operator::ExpOp, other; shift=nothing, kwargs...)
    value = _operator_dense_or_self(other)
    transposed_operator = transpose(operator)
    return value isa AbstractVector ?
        apply(transposed_operator, value; shift, kwargs...) :
        begin
            result = apply(
                transposed_operator,
                transpose(value);
                shift,
                kwargs...,
            )
            result isa AbstractArray && ndims(result) == 3 ?
                permutedims(result, (2, 1, 3)) :
                result isa AbstractMatrix ? transpose(result) :
                (transpose(state) for state in result)
        end
end

function sandwich(operator::ExpOp, other; shift=nothing, kwargs...)
    value = _operator_dense_or_self(other)
    value isa AbstractMatrix && size(value, 1) == size(value, 2) ||
        throw(ArgumentError("sandwiched value must be a square matrix"))
    left = apply(operator, value; shift, kwargs...)
    if operator.grid === nothing
        return right_apply(
            adjoint(operator),
            left;
            shift=shift === nothing ? nothing : conj(shift),
            kwargs...,
        )
    end
    if operator.iterate
        return (
            right_apply(
                ExpOp(adjoint(operator.O); a=conj(operator.a) * scale),
                left_state;
                shift=shift === nothing ? nothing : conj(shift),
                kwargs...,
            )
            for (left_state, scale) in zip(left, operator.grid)
        )
    end
    states = [
        begin
            single = ExpOp(adjoint(operator.O); a=conj(operator.a) * scale)
            right_apply(
                single,
                @view(left[:, :, index]);
                shift=shift === nothing ? nothing : conj(shift),
                kwargs...,
            )
        end
        for (index, scale) in pairs(operator.grid)
    ]
    return cat(states...; dims=3)
end

Base.:*(operator::ExpOp, other) = apply(operator, other)

end
