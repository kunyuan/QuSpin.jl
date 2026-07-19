module Operators

using Arpack
using LinearAlgebra
using Serialization
using SparseArrays
using ..Basis: SpinBasis1D

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
    for coupling in normalized
        length(coupling) == length(op) + 1 ||
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
const NativeMatrix{T} = Union{Matrix{T},SparseMatrixCSC{T,Int}}

mutable struct Hamiltonian{T<:Number}
    basis::SpinBasis1D
    terms::Vector{OperatorTerm}
    data::NativeMatrix{T}
    dynamic_terms::Vector{Tuple{NativeMatrix{T},Any,Tuple}}
end

function _normalize_matrix_format(format; default=:dense)
    format === nothing && return default
    normalized = Symbol(lowercase(String(format)))
    normalized === :sparse && return :csc
    normalized in (:dense, :csc) && return normalized
    normalized in (:csr, :dia) && throw(
        ArgumentError(
            "matrix format '$normalized' is not a native Julia storage format; " *
            "use :csc for sparse storage or :dense",
        ),
    )
    throw(ArgumentError("matrix format must be :dense or :csc"))
end

function _matrix_with_format(
    data::AbstractMatrix,
    ::Type{T},
    format,
;
    copy_data::Bool=true,
) where {T<:Number}
    normalized = _normalize_matrix_format(format)
    if normalized === :dense
        return !copy_data && data isa Matrix{T} ? data : Matrix{T}(data)
    end
    if !copy_data && data isa SparseMatrixCSC{T,Int}
        return data
    end
    return SparseMatrixCSC{T,Int}(sparse(data))
end

_storage_format(data::Matrix) = :dense
_storage_format(data::SparseMatrixCSC) = :csc

function _hamiltonian_from_data(
    basis::SpinBasis1D,
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

function _assemble(
    basis::SpinBasis1D,
    terms::AbstractVector{<:OperatorTerm},
    ::Type{T},
    format=:dense,
) where {T}
    rows = Int[]
    columns = Int[]
    values = T[]
    for (column, initial_state) in pairs(basis.encoded_states)
        for term in terms, coupling in term.couplings
            amplitude = convert(T, first(coupling))
            state = initial_state
            alive = true
            for (op, site) in zip(term.op, coupling[2:end])
                state, factor, alive = _apply_local(basis, state, op, site)
                alive || break
                amplitude *= factor
            end
            alive || continue
            row = get(basis.lookup, state, 0)
            iszero(row) && continue
            push!(rows, row)
            push!(columns, column)
            push!(values, amplitude)
        end
    end
    normalized = _normalize_matrix_format(format)
    if normalized === :csc
        return sparse(rows, columns, values, length(basis), length(basis))
    end
    matrix = zeros(T, length(basis), length(basis))
    for (row, column, value) in zip(rows, columns, values)
        matrix[row, column] += value
    end
    return matrix
end

function Hamiltonian(
    basis::SpinBasis1D,
    terms::AbstractVector{<:OperatorTerm};
    static_fmt=:dense,
)
    normalized = OperatorTerm[terms...]
    T = _coefficient_type(normalized)
    return Hamiltonian{T}(
        basis,
        normalized,
        _assemble(basis, normalized, T, static_fmt),
        Tuple{NativeMatrix{T},Any,Tuple}[],
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
    selected_basis isa SpinBasis1D ||
        throw(ArgumentError("the current Hamiltonian backend requires a SpinBasis1D"))
    terms, matrices = _normalize_operator_terms(static_list)
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
    return Hamiltonian{dtype}(
        selected_basis,
        terms,
        static_matrix,
        dynamic_terms,
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
LinearAlgebra.ishermitian(H::Hamiltonian) = ishermitian(H.data)

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
    return Hamiltonian{T}(H.basis, copy(H.terms), static_matrix, dynamic)
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
aslinearoperator(H::Hamiltonian; time=0) =
    isempty(H.dynamic_terms) && iszero(time) ? H : _matrix_at(H, time)
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
    return Hamiltonian{T}(H.basis, Base.copy(H.terms), static_matrix, dynamic)
end
diagonal(H::Hamiltonian; time=0) = diag(_matrix_at(H, time))
toarray(H::Hamiltonian; time=0, order=nothing, out=nothing) =
    _copy_or_write(Matrix(_matrix_at(H, time)), out)
todense(H::Hamiltonian; kwargs...) = toarray(H; kwargs...)
tocsc(H::Hamiltonian; time=0) = SparseMatrixCSC(_matrix_at(H, time))
function tocsr(H::Hamiltonian; time=0)
    throw(
        ArgumentError(
            "CSR storage is not provided by Julia's SparseArrays stdlib; " *
            "use tocsc(H; time) for native sparse storage",
        ),
    )
end

function _copy_or_write(value, out)
    out === nothing && return copy(value)
    axes(out) == axes(value) ||
        throw(DimensionMismatch("out must have the same axes as the result"))
    copyto!(out, value)
    return out
end

function apply(
    H::Hamiltonian,
    value;
    time=0,
    check::Bool=true,
    out=nothing,
    overwrite_out::Bool=true,
    a::Number=1,
)
    result = a * (_matrix_at(H, time) * value)
    out === nothing && return result
    axes(out) == axes(result) ||
        throw(DimensionMismatch("out must have the same axes as the result"))
    overwrite_out ? copyto!(out, result) : (out .+= result)
    return out
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
    matrix = _matrix_at(H, time)
    result = value isa AbstractVector ?
        a .* vec(transpose(value) * matrix) :
        a .* (value * matrix)
    out === nothing && return result
    overwrite_out ? copyto!(out, result) : (out .+= result)
    return out
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
    matrix = _matrix_at(H, time)
    n = size(matrix, 1)
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
    target = ishermitian(matrix) ? Hermitian(matrix) : matrix
    values, vectors, _, _, _, _ = Arpack.eigs(
        target;
        nev=k,
        which=arpack_which,
        kwargs...,
    )
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
    ishermitian(matrix) && (ordered_values = real.(ordered_values))
    return return_eigenvectors ?
        (ordered_values, vectors[:, order]) :
        ordered_values
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
        for target in times
            target >= current ||
                throw(ArgumentError("times must be sorted and not precede t0"))
            steps = max(1, ceil(Int, (target - current) / max_step))
            step = (target - current) / steps
            derivative = if state isa AbstractVector
                (time, value) -> begin
                    matrix = _matrix_at(H, time)
                    imag_time ? -(matrix * value) : -im .* (matrix * value)
                end
            else
                (time, value) -> begin
                    matrix = _matrix_at(H, time)
                    imag_time ?
                        -(matrix * value + value * matrix) / 2 :
                        -im .* (matrix * value - value * matrix)
                end
            end
            for _ in 1:steps
                k1 = derivative(current, state)
                k2 = derivative(current + step / 2, state + step * k1 / 2)
                k3 = derivative(current + step / 2, state + step * k2 / 2)
                k4 = derivative(current + step, state + step * k3)
                state = state + step * (k1 + 2k2 + 2k3 + k4) / 6
                current += step
            end
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

    E, V = eigh(H)
    offsets = collect(times) .- t0
    if v0 isa AbstractVector
        coefficients = V' * v0
        states = [
            begin
                exponent = imag_time ? -time .* E : -im * time .* E
                state = V * (exp.(exponent) .* coefficients)
                imag_time ? state / norm(state) : state
            end
            for time in offsets
        ]
        iterate && return (state for state in states)
        return reduce(hcat, states)
    end
    size(v0, 1) == size(v0, 2) == size(H, 1) ||
        throw(DimensionMismatch("density matrix must match Hamiltonian"))
    states = [
        begin
            U = V * Diagonal(exp.(-im * time .* E)) * V'
            U * v0 * U'
        end
        for time in offsets
    ]
    iterate && return (state for state in states)
    return cat(states...; dims=3)
end

function expt_value(H::Hamiltonian, state; time=0, enforce_pure::Bool=false, kwargs...)
    matrix = _matrix_at(H, time)
    if state isa AbstractVector
        return dot(state, matrix * state)
    elseif ndims(state) == 3
        return [tr(@view(state[:, :, index]) * matrix) for index in axes(state, 3)]
    elseif enforce_pure || size(state, 2) != size(H, 1)
        return [dot(column, matrix * column) for column in eachcol(state)]
    end
    return tr(state * matrix)
end

function matrix_ele(
    H::Hamiltonian,
    left,
    right;
    time=0,
    diagonal::Bool=false,
    check::Bool=true,
)
    elements = left' * _matrix_at(H, time) * right
    return diagonal && elements isa AbstractMatrix ? diag(elements) : elements
end

function project_to(H::Hamiltonian, projector)
    P = Matrix(projector)
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
    matrix = _matrix_at(H, time)
    mean = expt_value(H, state; time, enforce_pure)
    H2 = _hamiltonian_from_data(H.basis, matrix * matrix)
    second = expt_value(H2, state; time, enforce_pure)
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

Matrix-free-facing operator protocol backed by the same native Julia term
assembly as `Hamiltonian`. The dense matrix is materialized once in this first
implementation; callers use `matvec`/`apply` rather than depending on storage.
"""
mutable struct QuantumLinearOperator{T<:Number}
    basis::SpinBasis1D
    static_list::Vector{OperatorTerm}
    base_data::Matrix{T}
    diagonal::Vector{T}
end

function QuantumLinearOperator(
    basis::SpinBasis1D,
    static_list::AbstractVector{<:OperatorTerm};
    diagonal=nothing,
)
    H = Hamiltonian(basis, static_list)
    T = eltype(H)
    diagonal_values = diagonal === nothing ?
        zeros(T, length(basis)) :
        Vector{T}(diagonal)
    length(diagonal_values) == length(basis) ||
        throw(DimensionMismatch("diagonal must have length Ns"))
    return QuantumLinearOperator{T}(
        basis,
        OperatorTerm[static_list...],
        H.data,
        diagonal_values,
    )
end

_linear_data(operator::QuantumLinearOperator) =
    operator.base_data + Diagonal(operator.diagonal)

Base.Matrix(operator::QuantumLinearOperator) = Matrix(_linear_data(operator))
Base.size(operator::QuantumLinearOperator) = size(operator.base_data)
Base.size(operator::QuantumLinearOperator, dimension::Integer) =
    size(operator.base_data, dimension)
Base.eltype(operator::QuantumLinearOperator) = eltype(operator.base_data)
Base.:*(operator::QuantumLinearOperator, value::AbstractVecOrMat) =
    _linear_data(operator) * value

function Base.getproperty(operator::QuantumLinearOperator, name::Symbol)
    name === :H && return adjoint(operator)
    name === :T && return transpose(operator)
    name === :Ns && return size(getfield(operator, :base_data), 1)
    name === :dtype && return eltype(getfield(operator, :base_data))
    name === :get_shape && return size(getfield(operator, :base_data))
    name === :ndim && return 2
    name === :shape && return size(getfield(operator, :base_data))
    return getfield(operator, name)
end

function _linear_from_matrix(
    source::QuantumLinearOperator,
    matrix::AbstractMatrix,
)
    T = eltype(matrix)
    return QuantumLinearOperator{T}(
        source.basis,
        copy(source.static_list),
        Matrix{T}(matrix),
        zeros(T, size(matrix, 1)),
    )
end

Base.copy(operator::QuantumLinearOperator) = deepcopy(operator)
Base.transpose(operator::QuantumLinearOperator) =
    _linear_from_matrix(operator, transpose(_linear_data(operator)))
Base.conj(operator::QuantumLinearOperator) =
    _linear_from_matrix(operator, conj(_linear_data(operator)))
Base.adjoint(operator::QuantumLinearOperator) =
    _linear_from_matrix(operator, adjoint(_linear_data(operator)))
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
    result = a * (_linear_data(operator) * value)
    return _copy_or_write(result, out)
end

function right_apply(
    operator::QuantumLinearOperator,
    value;
    out=nothing,
    a::Number=1,
    kwargs...,
)
    result = value isa AbstractVector ?
        a .* vec(transpose(value) * _linear_data(operator)) :
        a .* (value * _linear_data(operator))
    return _copy_or_write(result, out)
end

function eigsh(
    operator::QuantumLinearOperator;
    k::Integer=min(6, size(operator, 1)),
    which=:SA,
    kwargs...,
)
    H = _hamiltonian_from_data(operator.basis, _linear_data(operator))
    return eigsh(H; k, which, kwargs...)
end

function expt_value(
    operator::QuantumLinearOperator,
    state;
    enforce_pure::Bool=false,
    kwargs...,
)
    H = _hamiltonian_from_data(operator.basis, _linear_data(operator))
    return expt_value(H, state; enforce_pure, kwargs...)
end

function matrix_ele(
    operator::QuantumLinearOperator,
    left,
    right;
    diagonal::Bool=false,
    kwargs...,
)
    elements = left' * _linear_data(operator) * right
    return diagonal && elements isa AbstractMatrix ? diag(elements) : elements
end

function quant_fluct(
    operator::QuantumLinearOperator,
    state;
    enforce_pure::Bool=false,
    kwargs...,
)
    H = _hamiltonian_from_data(operator.basis, _linear_data(operator))
    return quant_fluct(H, state; enforce_pure, kwargs...)
end

"""
    QuantumOperator(basis, input_dict)

Parameter-dependent operator `sum(pars[key] * input_dict[key])`. Dictionary
values may be native `OperatorTerm` vectors or square matrices.
"""
mutable struct QuantumOperator{T<:Number}
    basis::SpinBasis1D
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
    basis::SpinBasis1D,
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
    throw(
        ArgumentError(
            "CSR storage is not provided by Julia's SparseArrays stdlib; " *
            "use tocsc(operator; pars) for native sparse storage",
        ),
    )
end
diagonal(operator::QuantumOperator; pars::AbstractDict=Dict()) =
    diag(_parameter_matrix(operator, pars))
LinearAlgebra.tr(operator::QuantumOperator; pars::AbstractDict=Dict()) =
    tr(_parameter_matrix(operator, pars))

function apply(
    operator::QuantumOperator,
    value;
    pars::AbstractDict=Dict(),
    out=nothing,
    overwrite_out::Bool=true,
    a::Number=1,
    kwargs...,
)
    result = a * (_parameter_matrix(operator, pars) * value)
    out === nothing && return result
    overwrite_out ? copyto!(out, result) : (out .+= result)
    return out
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
    matrix = _parameter_matrix(operator, pars)
    result = value isa AbstractVector ?
        a .* vec(transpose(value) * matrix) :
        a .* (value * matrix)
    out === nothing && return result
    overwrite_out ? copyto!(out, result) : (out .+= result)
    return out
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
    linear = QuantumLinearOperator(
        operator.basis,
        OperatorTerm[];
        diagonal=zeros(eltype(operator), size(operator, 1)),
    )
    return _linear_from_matrix(
        linear,
        _parameter_matrix(operator, pars),
    )
end

function expt_value(
    operator::QuantumOperator,
    state;
    pars::AbstractDict=Dict(),
    enforce_pure::Bool=false,
    kwargs...,
)
    return expt_value(
        tohamiltonian(operator; pars),
        state;
        enforce_pure,
        kwargs...,
    )
end

function matrix_ele(
    operator::QuantumOperator,
    left,
    right;
    pars::AbstractDict=Dict(),
    diagonal::Bool=false,
    kwargs...,
)
    elements = left' * _parameter_matrix(operator, pars) * right
    return diagonal && elements isa AbstractMatrix ? diag(elements) : elements
end

function quant_fluct(
    operator::QuantumOperator,
    state;
    pars::AbstractDict=Dict(),
    enforce_pure::Bool=false,
    kwargs...,
)
    return quant_fluct(
        tohamiltonian(operator; pars),
        state;
        enforce_pure,
        kwargs...,
    )
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
mutable struct ExpOp
    O::Any
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
    matrix = _operator_dense(O)
    size(matrix, 1) == size(matrix, 2) ||
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
    name === :Ns && return size(_operator_dense(getfield(operator, :O)), 1)
    name === :get_shape && return size(_operator_dense(getfield(operator, :O)))
    name === :ndim && return 2
    return getfield(operator, name)
end

Base.size(operator::ExpOp) = operator.get_shape
Base.copy(operator::ExpOp) = deepcopy(operator)
isexp_op(value) = value isa ExpOp

function Base.transpose(operator::ExpOp)
    return _copy_exp_with_operator(operator, transpose(_operator_dense(operator.O)), operator.a)
end

function Base.conj(operator::ExpOp)
    return _copy_exp_with_operator(operator, conj(_operator_dense(operator.O)), conj(operator.a))
end

function Base.adjoint(operator::ExpOp)
    return _copy_exp_with_operator(operator, adjoint(_operator_dense(operator.O)), conj(operator.a))
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
    shifted = shift === nothing ?
        matrix :
        matrix + shift * Matrix{promote_type(eltype(matrix), typeof(shift))}(
            I,
            size(matrix)...,
        )
    return exp((operator.a * scale) * shifted)
end

get_mat(operator::ExpOp; dense::Bool=true, kwargs...) = _exp_matrix(operator)

function _grid_results(operator::ExpOp, operation; shift=nothing)
    if operator.grid === nothing
        return operation(_exp_matrix(operator; shift))
    end
    results = [
        operation(_exp_matrix(operator, scale; shift))
        for scale in operator.grid
    ]
    operator.iterate && return (result for result in results)
    first(results) isa AbstractVector && return reduce(hcat, results)
    return cat(results...; dims=3)
end

function apply(operator::ExpOp, other; shift=nothing, kwargs...)
    return _grid_results(
        operator,
        matrix -> matrix * _operator_dense_or_self(other);
        shift,
    )
end

_operator_dense_or_self(value::Hamiltonian) = Matrix(value)
_operator_dense_or_self(value) = value

function right_apply(operator::ExpOp, other; shift=nothing, kwargs...)
    value = _operator_dense_or_self(other)
    return _grid_results(
        operator,
        matrix -> value isa AbstractVector ?
            vec(transpose(value) * matrix) :
            value * matrix;
        shift,
    )
end

function sandwich(operator::ExpOp, other; shift=nothing, kwargs...)
    value = _operator_dense_or_self(other)
    value isa AbstractMatrix && size(value, 1) == size(value, 2) ||
        throw(ArgumentError("sandwiched value must be a square matrix"))
    return _grid_results(operator, matrix -> matrix * value * matrix'; shift)
end

Base.:*(operator::ExpOp, other) = apply(operator, other)

end
