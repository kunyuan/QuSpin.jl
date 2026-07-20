"""
    UserBasis(dtype, N, op_dict; sps=2, pre_check_state=nothing, ...)

User-defined finite basis. Local operators may be supplied as `sps × sps`
matrices or deterministic callbacks `(encoded_state, site) -> (state, factor)`.
"""
struct UserBasis <: AbstractBasis
    base::DiscreteBasis{:user}
    basis_dtype::DataType
    op_dict::Dict{Any,Any}
    allowed_ops::Tuple
    user_blocks::Dict{Symbol,Any}
    user_noncommuting_bits::Any
end

function UserBasis(
    basis_dtype::Type,
    N::Integer,
    op_dict::AbstractDict;
    sps::Integer=2,
    pcon_dict=nothing,
    pre_check_state=nothing,
    allowed_ops=nothing,
    parallel::Bool=false,
    Ns_block_est=nothing,
    _make_basis::Bool=true,
    block_order=nothing,
    noncommuting_bits=Any[],
    _Np=nothing,
    states=nothing,
    blocks...,
)
    basis_dtype <: Integer || throw(ArgumentError("basis_dtype must be an integer type"))
    explicit_states = states
    if explicit_states === nothing && pcon_dict isa AbstractDict
        explicit_states = get(pcon_dict, :states, get(pcon_dict, "states", nothing))
    end
    predicate = if pre_check_state === nothing
        state -> true
    elseif applicable(pre_check_state, zero(UInt64), N)
        state -> Bool(pre_check_state(state, N))
    elseif applicable(pre_check_state, zero(UInt64))
        state -> Bool(pre_check_state(state))
    else
        throw(ArgumentError(
            "pre_check_state must accept (state) or (state, N)",
        ))
    end
    operators =
        allowed_ops === nothing ?
        Tuple(keys(op_dict)) :
        Tuple(allowed_ops)
    base = if explicit_states === nothing
        keep = occupations -> begin
            encoded = UInt64(sum(
                occupations[site] * sps^(site - 1)
                for site in 1:N
            ))
            predicate(encoded)
        end
        _make_discrete_basis(
            Val(:user),
            N,
            sps,
            pcon_dict,
            keep,
            "user-defined finite basis",
            operators,
        )
    else
        dimension = BigInt(sps)^N
        dimension <= typemax(UInt64) ||
            throw(ArgumentError("basis encoding exceeds UInt64"))
        encoded = sort!(unique(UInt64.(collect(explicit_states))))
        filter!(
            state -> state < UInt64(dimension) && predicate(state),
            encoded,
        )
        _discrete_basis_from_encoded(
            Val(:user),
            N,
            sps,
            pcon_dict,
            encoded,
            "user-defined finite basis",
            operators,
        )
    end
    return UserBasis(
        base,
        basis_dtype,
        Dict{Any,Any}(op_dict),
        operators,
        Dict{Symbol,Any}(Symbol(key) => value for (key, value) in blocks),
        noncommuting_bits,
    )
end

Base.length(basis::UserBasis) = length(basis.base)

function Base.getproperty(basis::UserBasis, name::Symbol)
    name === :N && return getfield(basis, :base).L
    name === :Ns && return length(basis)
    name === :sps && return getfield(basis, :base).sps
    name === :states && return getfield(basis, :basis_dtype).(
        getfield(basis, :base).encoded_states,
    )
    name === :blocks && return getfield(basis, :user_blocks)
    name === :description && return "user-defined finite basis"
    name === :dtype && return getfield(basis, :basis_dtype)
    name === :noncommuting_bits && return getfield(basis, :user_noncommuting_bits)
    name === :operators && return getfield(basis, :allowed_ops)
    return getfield(basis, name)
end

states(basis::UserBasis) =
    basis.basis_dtype.(basis.base.encoded_states)
projection_matrix(
    basis::UserBasis,
    ::Type{T}=Float64;
    sparse::Bool=false,
) where {T<:Number} =
    projection_matrix(basis.base, T; sparse)
project_from(basis::UserBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis.base, vector; kwargs...)
get_vec(basis::UserBasis, vector::AbstractVecOrMat; kwargs...) =
    get_vec(basis.base, vector; kwargs...)
expanded_form(basis::UserBasis, static=Any[], dynamic=Any[]) =
    (static, dynamic)
state_index(basis::UserBasis, state::Integer) = state_index(basis.base, state)
state_at(basis::UserBasis, index::Integer) =
    basis.basis_dtype(state_at(basis.base, index))
int_to_state(basis::UserBasis, state::Integer; kwargs...) =
    int_to_state(basis.base, state; kwargs...)
state_to_int(basis::UserBasis, state::AbstractString) =
    basis.basis_dtype(state_to_int(basis.base, state))
partial_trace(basis::UserBasis, state::AbstractVecOrMat; kwargs...) =
    partial_trace(basis.base, state; kwargs...)
ent_entropy(basis::UserBasis, state::AbstractVecOrMat; kwargs...) =
    ent_entropy(basis.base, state; kwargs...)

function _user_operator(basis::UserBasis, op)
    haskey(basis.op_dict, op) && return basis.op_dict[op]
    haskey(basis.op_dict, string(op)) && return basis.op_dict[string(op)]
    throw(ArgumentError("operator '$op' is not defined"))
end

function operator_matrix(
    basis::UserBasis,
    opstring::AbstractString,
    couplings,
    ;
    sparse::Bool=false,
)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    for coupling in couplings
        length(coupling) == length(opstring) + 1 ||
            throw(ArgumentError("operator arity and sites differ"))
        for (column, initial) in pairs(basis.base.encoded_states)
            branches = Dict(initial => complex(first(coupling)))
            for (op, site) in Iterators.reverse(
                collect(zip(opstring, coupling[2:end])),
            )
                1 <= site <= basis.N ||
                    throw(ArgumentError("site must lie in 1:$(basis.N)"))
                definition = _user_operator(basis, op)
                next_branches = Dict{UInt64,ComplexF64}()
                for (encoded, amplitude) in branches
                    if definition isa AbstractMatrix
                        size(definition) == (basis.sps, basis.sps) ||
                            throw(DimensionMismatch("local operator must have shape (sps,sps)"))
                        occupations = _digits(encoded, basis.N, basis.sps)
                        old = occupations[site]
                        for new in 0:(basis.sps - 1)
                            factor = definition[new + 1, old + 1]
                            iszero(factor) && continue
                            updated = UInt64(
                                Int(encoded) +
                                (new - old) * basis.sps^(site - 1),
                            )
                            next_branches[updated] =
                                get(next_branches, updated, 0) + amplitude * factor
                        end
                    elseif definition isa Function
                        updated, factor = definition(encoded, site)
                        next_branches[UInt64(updated)] =
                            get(next_branches, UInt64(updated), 0) + amplitude * factor
                    else
                        throw(ArgumentError("operators must be matrices or callbacks"))
                    end
                end
                branches = next_branches
            end
            for (encoded, amplitude) in branches
                row = get(basis.base.lookup, encoded, 0)
                row == 0 && continue
                push!(rows, row)
                push!(columns, column)
                push!(values, amplitude)
            end
        end
    end
    matrix = SparseArrays.sparse(
        rows,
        columns,
        values,
        length(basis),
        length(basis),
    )
    return sparse ? matrix : Matrix(matrix)
end

function inplace_op!(out, basis::UserBasis, opstring, couplings)
    size(out) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("out must have shape (Ns,Ns)"))
    out .+= operator_matrix(
        basis,
        opstring,
        couplings;
        sparse=true,
    )
    return out
end

op_bra_ket(basis::UserBasis, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)
