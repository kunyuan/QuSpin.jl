"""
    UserBasis(dtype, N, op_dict; sps=2, pre_check_state=nothing, ...)

User-defined finite basis. Local operators may be supplied as `sps × sps`
matrices or deterministic callbacks `(encoded_state, site) -> (state, factor)`.
"""
struct UserBasis{T<:Integer,A<:Tuple,N} <: AbstractBasis
    base::DiscreteBasis{:user}
    basis_dtype::Type{T}
    op_dict::Dict{Any,Any}
    allowed_ops::A
    user_blocks::Dict{Symbol,Any}
    user_noncommuting_bits::N
end

"""
    constraint_states(N; sps=2, prefix_allowed, state_allowed, dtype=UInt64)

Enumerate finite-product states without scanning the full `sps^N` space.
`prefix_allowed(occupations, site)` is called while a prefix is being built
and may prune the entire remaining branch. `state_allowed(occupations)` is
called only for complete states. Occupations use the same convention as the
built-in discrete bases: site one is the least-significant encoded digit.

The returned encoded states can be passed directly to `UserBasis(...;
states=...)`. This is intended for reusable local constraints such as Rydberg
blockade, hard local exclusions, and finite-state automata; it does not attach
model-specific operator semantics.
"""
function constraint_states(
    N::Integer;
    sps::Integer=2,
    prefix_allowed=(occupations, site) -> true,
    state_allowed=occupations -> true,
    dtype::Type{T}=UInt64,
) where {T<:Unsigned}
    N > 0 || throw(ArgumentError("N must be positive"))
    sps > 0 || throw(ArgumentError("sps must be positive"))
    maximum_state = BigInt(sps)^N - 1
    maximum_state <= typemax(T) ||
        throw(ArgumentError("state encoding exceeds $T"))

    occupations = zeros(Int, N)
    weights = T[T(sps)^(site - 1) for site in 1:N]
    encoded = T[]

    function append_prefix!(site::Int, value::T)
        if site > N
            state_allowed(occupations) && push!(encoded, value)
            return
        end
        for occupation in 0:(sps - 1)
            occupations[site] = occupation
            prefix_allowed(occupations, site) || continue
            append_prefix!(
                site + 1,
                value + T(occupation) * weights[site],
            )
        end
        occupations[site] = 0
        return
    end

    append_prefix!(1, zero(T))
    sort!(encoded)
    return encoded
end

_user_dict_get(dictionary::AbstractDict, key::Symbol, default=nothing) =
    haskey(dictionary, key) ?
    dictionary[key] :
    get(dictionary, String(key), default)

function _next_user_pcon_state(next_state, state, counter, N, arguments)
    if applicable(next_state, state, counter, N, arguments)
        return UInt64(next_state(state, counter, N, arguments))
    elseif applicable(next_state, state, counter, N)
        return UInt64(next_state(state, counter, N))
    elseif applicable(next_state, state, counter)
        return UInt64(next_state(state, counter))
    end
    throw(ArgumentError(
        "pcon_dict next_state must accept (state,counter,N,args), " *
        "(state,counter,N), or (state,counter)",
    ))
end

function _user_pcon_states(pcon_dict::AbstractDict, N::Int, predicate)
    next_state = _user_dict_get(pcon_dict, :next_state)
    get_dimension = _user_dict_get(pcon_dict, :get_Ns_pcon)
    get_initial = _user_dict_get(pcon_dict, :get_s0_pcon)
    sectors = _user_dict_get(pcon_dict, :Np)
    any(value -> value === nothing, (next_state, get_dimension, get_initial, sectors)) &&
        return nothing
    arguments = _user_dict_get(pcon_dict, :next_state_args, ())
    selected_sectors = sectors isa Union{Integer,Tuple} ?
        (sectors,) :
        Tuple(sectors)
    encoded = UInt64[]
    for sector in selected_sectors
        count = Int(get_dimension(N, sector))
        count >= 0 ||
            throw(ArgumentError("get_Ns_pcon must return a nonnegative size"))
        iszero(count) && continue
        state = UInt64(get_initial(N, sector))
        for counter in 0:(count - 1)
            predicate(state) && push!(encoded, state)
            counter == count - 1 && break
            state = _next_user_pcon_state(
                next_state,
                state,
                counter,
                N,
                arguments,
            )
        end
    end
    sort!(unique!(encoded))
    return encoded
end

function _call_user_symmetry(callback, state::UInt64, N::Int, arguments)
    result = if applicable(callback, state, N, arguments)
        callback(state, N, arguments)
    elseif applicable(callback, state, N)
        callback(state, N)
    elseif applicable(callback, state, arguments)
        callback(state, arguments)
    elseif applicable(callback, state)
        callback(state)
    else
        throw(ArgumentError(
            "a user symmetry callback must accept state and optional N/arguments",
        ))
    end
    if result isa Tuple
        length(result) == 2 ||
            throw(ArgumentError("a symmetry callback tuple must be (state, phase)"))
        return UInt64(result[1]), ComplexF64(result[2])
    end
    return UInt64(result), 1.0 + 0im
end

function _user_symmetry_basis(
    basis::DiscreteBasis{:user},
    block_order,
    blocks,
)
    isempty(blocks) && return basis, Dict{Symbol,Any}()
    block_dictionary = Dict{Symbol,Any}(
        Symbol(name) => value for (name, value) in blocks
    )
    ordered_names = block_order === nothing ?
        Symbol[Symbol(name) for (name, _) in blocks] :
        Symbol.(collect(block_order))
    for name in keys(block_dictionary)
        name in ordered_names || push!(ordered_names, name)
    end
    all(name -> haskey(block_dictionary, name), ordered_names) ||
        throw(ArgumentError("block_order refers to an unknown symmetry block"))

    parent_states = basis.encoded_states
    projector = nothing
    metadata = Dict{Symbol,Any}()
    symmetry_blocks = copy(basis.symmetry.blocks)
    for name in ordered_names
        specification = block_dictionary[name]
        (specification isa Tuple || specification isa AbstractVector) ||
            throw(ArgumentError("user symmetry blocks must be tuples"))
        transform, period, quantum_number = if length(specification) >= 3 &&
                                               specification[1] isa Function
            callback = specification[1]
            selected_period = Int(specification[2])
            selected_period > 0 ||
                throw(ArgumentError("symmetry period must be positive"))
            selected_quantum_number = Int(specification[3])
            arguments = length(specification) >= 4 ?
                specification[4] :
                ()
            (
                state -> _call_user_symmetry(
                    callback,
                    state,
                    basis.L,
                    arguments,
                ),
                selected_period,
                selected_quantum_number,
            )
        elseif length(specification) >= 2
            selected_transform, selected_period =
                _general_discrete_transform(basis, specification[1])
            selected_transform, selected_period, Int(specification[2])
        else
            throw(ArgumentError(
                "user symmetry blocks must be (callback, period, q, args) " *
                "or (map, q)",
            ))
        end
        eigenvalue =
            cis(2π * mod(quantum_number, period) / period)
        projector = if projector === nothing
            _cyclic_projector(
                parent_states,
                basis.lookup,
                transform,
                ComplexF64(eigenvalue),
            )
        else
            _intersect_eigenspace(
                projector,
                _signed_permutation(parent_states, basis.lookup, transform),
                ComplexF64(eigenvalue),
            )
        end
        symmetry_blocks[name] = mod(quantum_number, period)
        symmetry_blocks[Symbol(name, :_period)] = period
        metadata[name] = mod(quantum_number, period)
        metadata[Symbol(name, :_period)] = period
    end
    symmetry_blocks[:block_order] = copy(ordered_names)
    metadata[:block_order] = copy(ordered_names)
    symmetry = _finalize_symmetry_data(
        parent_states,
        projector,
        symmetry_blocks;
        parent_occupations=basis.occupations,
    )
    representatives =
        _representative_states(parent_states, symmetry.projector)
    occupations = isempty(representatives) ?
        Matrix{Int}(undef, 0, basis.L) :
        reduce(vcat, permutedims.(
            _digits(state, basis.L, basis.sps)
            for state in representatives
        ))
    reduced = DiscreteBasis{:user}(
        basis.L,
        basis.sps,
        basis.conservation,
        representatives,
        occupations,
        Dict(state => index for (index, state) in pairs(representatives)),
        basis.description,
        basis.operators,
        symmetry,
    )
    return reduced, metadata
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
    explicit_states = states
    if explicit_states === nothing && pcon_dict isa AbstractDict
        explicit_states = _user_dict_get(pcon_dict, :states)
        explicit_states === nothing &&
            (explicit_states = _user_pcon_states(
                pcon_dict,
                Int(N),
                predicate,
            ))
    end
    operators =
        allowed_ops === nothing ?
        Tuple(keys(op_dict)) :
        Tuple(allowed_ops)
    if !_make_basis
        block_keywords = (; blocks...)
        builder = () -> UserBasis(
            basis_dtype,
            N,
            op_dict;
            sps,
            pcon_dict,
            pre_check_state,
            allowed_ops,
            parallel,
            Ns_block_est,
            _make_basis=true,
            block_order,
            noncommuting_bits,
            _Np,
            states,
            block_keywords...,
        )
        metadata = Dict{Symbol,Any}(
            :L => Int(N),
            :N => Int(N),
            :Ns => 1,
            :sps => Int(sps),
            :dtype => basis_dtype,
            :states => basis_dtype[0],
            :encoded_states => UInt64[0],
            :description => "deferred user-defined finite basis",
            :operators => operators,
            :noncommuting_bits => noncommuting_bits,
            :blocks => Dict{Symbol,Any}(
                Symbol(name) => value for (name, value) in blocks
            ),
        )
        metadata[:blocks][:parallel] = parallel
        Ns_block_est === nothing ||
            (metadata[:Ns_block_est] = Int(Ns_block_est))
        target_type = UserBasis{
            basis_dtype,
            typeof(operators),
            typeof(noncommuting_bits),
        }
        return _deferred_basis(target_type, builder, metadata)
    end
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
            parallel=parallel,
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
    reduction_blocks = Dict{Symbol,Any}(
        Symbol(name) => value
        for (name, value) in blocks
        if value isa Tuple || value isa AbstractVector
    )
    reduced_base, symmetry_metadata =
        _user_symmetry_basis(base, block_order, reduction_blocks)
    for (name, value) in blocks
        (value isa Tuple || value isa AbstractVector) && continue
        symmetry_metadata[Symbol(name)] = value
    end
    symmetry_metadata[:parallel] = parallel
    symmetry_metadata[:made_basis] = _make_basis
    Ns_block_est === nothing ||
        (symmetry_metadata[:Ns_block_est] = Int(Ns_block_est))
    return UserBasis(
        reduced_base,
        basis_dtype,
        Dict{Any,Any}(op_dict),
        operators,
        symmetry_metadata,
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
    pcon::Bool=false,
) where {T<:Number} =
    projection_matrix(basis.base, T; sparse, pcon)
_full_projection_dimension(basis::UserBasis) =
    _full_projection_dimension(basis.base)
_projection_output_dimension(basis::UserBasis, pcon::Bool) =
    _projection_output_dimension(basis.base, pcon)
_pcon_projection_matrix(basis::UserBasis, ::Type{T}) where {T<:Number} =
    _pcon_projection_matrix(basis.base, T)
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

# Compile dynamic user definitions once, then keep state-by-state assembly in
# concrete matrix/callback action types. The final assembler writes CSC columns
# directly and reuses its branch and accumulation buffers across all states.
struct _CompiledUserMatrixAction{T<:Number}
    weight::UInt64
    offsets::Vector{Int}
    deltas::Vector{Int128}
    factors::Vector{T}
    branching::Bool
end

struct _CompiledUserCallbackAction{F}
    site::Int
    definition::F
end

struct _CompiledUserTransition{T<:Number,A<:Tuple}
    coefficient::T
    actions::A
end

struct _CompiledUserTerm{T<:Number,A<:Tuple}
    transitions::Vector{_CompiledUserTransition{T,A}}
end

function _convert_user_coefficient(::Type{T}, coefficient) where {T<:Number}
    if T <: Real && coefficient isa Complex
        iszero(imag(coefficient)) ||
            throw(InexactError(:_user_operator_csc, T, coefficient))
        return convert(T, real(coefficient))
    end
    return convert(T, coefficient)
end

function _compile_user_action(
    definition::AbstractMatrix,
    site::Int,
    basis::UserBasis,
    ::Type{T},
) where {T<:Number}
    size(definition) == (basis.sps, basis.sps) ||
        throw(DimensionMismatch("local operator must have shape (sps,sps)"))
    weight = UInt64(basis.sps)^(site - 1)
    offsets = Vector{Int}(undef, basis.sps + 1)
    deltas = Int128[]
    factors = T[]
    offsets[1] = 1
    for old in 0:(basis.sps - 1)
        for new in 0:(basis.sps - 1)
            factor = definition[new + 1, old + 1]
            iszero(factor) && continue
            push!(
                deltas,
                Int128(new - old) * Int128(weight),
            )
            push!(factors, _convert_user_coefficient(T, factor))
        end
        offsets[old + 2] = length(factors) + 1
    end
    return _CompiledUserMatrixAction{T}(
        weight,
        offsets,
        deltas,
        factors,
        any(
            offsets[index + 1] - offsets[index] > 1
            for index in 1:basis.sps
        ),
    )
end

_compile_user_action(
    definition::Function,
    site::Int,
    ::UserBasis,
    ::Type{T},
) where {T<:Number} = _CompiledUserCallbackAction(site, definition)

function _compile_user_action(
    definition,
    ::Int,
    ::UserBasis,
    ::Type{T},
) where {T<:Number}
    throw(ArgumentError("operators must be matrices or callbacks"))
end

_compile_user_actions(
    ::UserBasis,
    ::Tuple{},
    ::Tuple{},
    ::Type{T},
) where {T<:Number} = ()

function _compile_user_actions(
    basis::UserBasis,
    operators::Tuple,
    sites::Tuple,
    ::Type{T},
) where {T<:Number}
    length(operators) == length(sites) ||
        throw(ArgumentError("operator arity and sites differ"))
    site = Int(first(sites))
    1 <= site <= basis.N ||
        throw(ArgumentError("site must lie in 1:$(basis.N)"))
    definition = _user_operator(basis, first(operators))
    action = _compile_user_action(definition, site, basis, T)
    remaining = _compile_user_actions(
        basis,
        Base.tail(operators),
        Base.tail(sites),
        T,
    )
    return (action, remaining...)
end

function _compile_user_transition(
    basis::UserBasis,
    opstring::AbstractString,
    coupling,
    ::Type{T},
) where {T<:Number}
    length(coupling) == length(opstring) + 1 ||
        throw(ArgumentError("operator arity and sites differ"))
    operators = reverse(Tuple(opstring))
    sites = reverse(Tuple(coupling[2:end]))
    actions = _compile_user_actions(basis, operators, sites, T)
    coefficient = _convert_user_coefficient(T, first(coupling))
    return _CompiledUserTransition{T,typeof(actions)}(
        coefficient,
        actions,
    )
end

function _append_user_transition(::Nothing, transition)
    transitions = Vector{typeof(transition)}()
    push!(transitions, transition)
    return transitions
end

function _append_user_transition(
    transitions::Vector{T},
    transition::T,
) where {T}
    push!(transitions, transition)
    return transitions
end

function _compile_user_term(
    basis::UserBasis,
    opstring::AbstractString,
    couplings,
    ::Type{T},
) where {T<:Number}
    transitions = nothing
    for coupling in couplings
        transition = _compile_user_transition(
            basis,
            opstring,
            coupling,
            T,
        )
        iszero(transition.coefficient) && continue
        transitions = _append_user_transition(transitions, transition)
    end
    transitions === nothing && return nothing
    return _CompiledUserTerm(transitions)
end

@inline function _apply_compiled_user_action!(
    output_states::Vector{UInt64},
    output_values::Vector{T},
    output_lookup::Dict{UInt64,Int},
    action::_CompiledUserMatrixAction{T},
    input_states::Vector{UInt64},
    input_values::Vector{T},
) where {T<:Number}
    action.branching && empty!(output_lookup)
    @inbounds for branch in eachindex(input_states)
        encoded = input_states[branch]
        old = Int((encoded ÷ action.weight) % UInt64(length(action.offsets) - 1))
        for pointer in action.offsets[old + 1]:(action.offsets[old + 2] - 1)
            amplitude = input_values[branch] * action.factors[pointer]
            iszero(amplitude) && continue
            updated = UInt64(
                Int128(encoded) + action.deltas[pointer],
            )
            if action.branching
                index = get(output_lookup, updated, 0)
                if iszero(index)
                    push!(output_states, updated)
                    push!(output_values, amplitude)
                    output_lookup[updated] = length(output_states)
                else
                    output_values[index] += amplitude
                end
            else
                push!(output_states, updated)
                push!(output_values, amplitude)
            end
        end
    end
    return nothing
end

@inline function _apply_compiled_user_action!(
    output_states::Vector{UInt64},
    output_values::Vector{T},
    ::Dict{UInt64,Int},
    action::_CompiledUserCallbackAction,
    input_states::Vector{UInt64},
    input_values::Vector{T},
) where {T<:Number}
    @inbounds for branch in eachindex(input_states)
        updated, factor = action.definition(
            input_states[branch],
            action.site,
        )
        amplitude = input_values[branch] *
            _convert_user_coefficient(T, factor)
        iszero(amplitude) && continue
        push!(output_states, UInt64(updated))
        push!(output_values, amplitude)
    end
    return nothing
end

@inline _apply_compiled_user_actions!(
    input_states::Vector{UInt64},
    input_values::Vector{T},
    output_states::Vector{UInt64},
    output_values::Vector{T},
    ::Dict{UInt64,Int},
    ::Tuple{},
) where {T<:Number} = (input_states, input_values)

@inline function _apply_compiled_user_actions!(
    input_states::Vector{UInt64},
    input_values::Vector{T},
    output_states::Vector{UInt64},
    output_values::Vector{T},
    output_lookup::Dict{UInt64,Int},
    actions::Tuple,
) where {T<:Number}
    empty!(output_states)
    empty!(output_values)
    _apply_compiled_user_action!(
        output_states,
        output_values,
        output_lookup,
        first(actions),
        input_states,
        input_values,
    )
    return _apply_compiled_user_actions!(
        output_states,
        output_values,
        input_states,
        input_values,
        output_lookup,
        Base.tail(actions),
    )
end

@inline function _accumulate_user_term_column!(
    touched::Vector{Int},
    markers::Vector{Int},
    accumulated::Vector{T},
    term::_CompiledUserTerm{T},
    initial::UInt64,
    column::Int,
    lookup::Dict{UInt64,Int},
    first_states::Vector{UInt64},
    first_values::Vector{T},
    second_states::Vector{UInt64},
    second_values::Vector{T},
    branch_lookup::Dict{UInt64,Int},
) where {T<:Number}
    for transition in term.transitions
        empty!(first_states)
        empty!(first_values)
        push!(first_states, initial)
        push!(first_values, transition.coefficient)
        branch_states, branch_values = _apply_compiled_user_actions!(
            first_states,
            first_values,
            second_states,
            second_values,
            branch_lookup,
            transition.actions,
        )
        @inbounds for branch in eachindex(branch_states)
            row = get(lookup, branch_states[branch], 0)
            iszero(row) && continue
            if markers[row] == column
                accumulated[row] += branch_values[branch]
            else
                markers[row] = column
                accumulated[row] = branch_values[branch]
                push!(touched, row)
            end
        end
    end
    return nothing
end

function _user_operator_csc(
    basis::UserBasis,
    specifications,
    ::Type{T}=ComplexF64,
) where {T<:Number}
    _has_symmetry(basis.base.symmetry) &&
        throw(ArgumentError(
            "direct UserBasis CSC assembly requires an unreduced basis",
        ))
    compiled_terms = Any[]
    for (opstring, couplings) in specifications
        term = _compile_user_term(
            basis,
            opstring,
            couplings,
            T,
        )
        term === nothing || push!(compiled_terms, term)
    end

    return _assemble_compiled_user_csc(
        basis,
        Tuple(compiled_terms),
        T,
    )
end

function _assemble_compiled_user_csc(
    basis::UserBasis,
    compiled_terms::Tuple,
    ::Type{T},
) where {T<:Number}
    dimension = length(basis)
    column_pointers = Vector{Int}(undef, dimension + 1)
    column_pointers[1] = 1
    row_values = Int[]
    nonzeros = T[]
    touched = Int[]
    markers = zeros(Int, dimension)
    accumulated = zeros(T, dimension)
    first_states = UInt64[]
    first_values = T[]
    second_states = UInt64[]
    second_values = T[]
    branch_lookup = Dict{UInt64,Int}()
    sizehint!(touched, length(compiled_terms))
    sizehint!(first_states, 16)
    sizehint!(first_values, 16)
    sizehint!(second_states, 16)
    sizehint!(second_values, 16)

    @inbounds for column in eachindex(basis.base.encoded_states)
        empty!(touched)
        initial = basis.base.encoded_states[column]
        for term in compiled_terms
            _accumulate_user_term_column!(
                touched,
                markers,
                accumulated,
                term,
                initial,
                column,
                basis.base.lookup,
                first_states,
                first_values,
                second_states,
                second_values,
                branch_lookup,
            )
        end
        sort!(touched)
        for row in touched
            value = accumulated[row]
            iszero(value) && continue
            push!(row_values, row)
            push!(nonzeros, value)
        end
        column_pointers[column + 1] = length(row_values) + 1
    end
    return SparseMatrixCSC{T,Int}(
        dimension,
        dimension,
        column_pointers,
        row_values,
        nonzeros,
    )
end

function _apply_user_definition!(
    next_branches::Dict{UInt64,ComplexF64},
    definition::AbstractMatrix,
    encoded::UInt64,
    amplitude,
    site::Int,
    sps::Int,
    weight::UInt64,
)
    size(definition) == (sps, sps) ||
        throw(DimensionMismatch("local operator must have shape (sps,sps)"))
    old = Int((encoded ÷ weight) % UInt64(sps))
    for new in 0:(sps - 1)
        factor = definition[new + 1, old + 1]
        iszero(factor) && continue
        updated = UInt64(
            Int128(encoded) + Int128(new - old) * Int128(weight),
        )
        next_branches[updated] =
            get(next_branches, updated, 0) + amplitude * factor
    end
    return next_branches
end

function _apply_user_definition!(
    next_branches::Dict{UInt64,ComplexF64},
    definition::Function,
    encoded::UInt64,
    amplitude,
    site::Int,
    sps::Int,
    weight::UInt64,
)
    updated, factor = definition(encoded, site)
    converted = UInt64(updated)
    next_branches[converted] =
        get(next_branches, converted, 0) + amplitude * factor
    return next_branches
end

function _apply_user_definition!(
    next_branches::Dict{UInt64,ComplexF64},
    definition,
    encoded::UInt64,
    amplitude,
    site::Int,
    sps::Int,
    weight::UInt64,
)
    throw(ArgumentError("operators must be matrices or callbacks"))
end

function _user_callback_entry(
    definition::Function,
    encoded::UInt64,
    site::Int,
)
    updated, factor = definition(encoded, site)
    return UInt64(updated), complex(factor)
end

function _user_operator_triplets(
    basis::UserBasis,
    opstring::AbstractString,
    couplings,
)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.N
    ]
    for coupling in couplings
        length(coupling) == length(opstring) + 1 ||
            throw(ArgumentError("operator arity and sites differ"))
        actions = [
            (_user_operator(basis, op), Int(site))
            for (op, site) in Iterators.reverse(
                collect(zip(opstring, coupling[2:end])),
            )
        ]
        for (_, site) in actions
            1 <= site <= basis.N ||
                throw(ArgumentError("site must lie in 1:$(basis.N)"))
        end
        deterministic = all(action -> action[1] isa Function, actions)
        for (column, initial) in pairs(basis.base.encoded_states)
            if deterministic
                encoded = initial
                amplitude = complex(first(coupling))
                for (definition, site) in actions
                    encoded, factor =
                        _user_callback_entry(definition, encoded, site)
                    amplitude *= factor
                end
                row = get(basis.base.lookup, encoded, 0)
                row == 0 && continue
                push!(rows, row)
                push!(columns, column)
                push!(values, amplitude)
                continue
            end
            branches = Dict(initial => complex(first(coupling)))
            for (definition, site) in actions
                next_branches = Dict{UInt64,ComplexF64}()
                for (encoded, amplitude) in branches
                    _apply_user_definition!(
                        next_branches,
                        definition,
                        encoded,
                        amplitude,
                        site,
                        basis.sps,
                        weights[site],
                    )
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
    return rows, columns, values
end

function operator_matrix(
    basis::UserBasis,
    opstring::AbstractString,
    couplings,
    ;
    sparse::Bool=false,
)
    if _has_symmetry(basis.base.symmetry)
        parent_base = _discrete_parent_basis(basis.base)
        parent = UserBasis(
            parent_base,
            basis.basis_dtype,
            basis.op_dict,
            basis.allowed_ops,
            Dict{Symbol,Any}(),
            basis.user_noncommuting_bits,
        )
        parent_matrix = operator_matrix(
            parent,
            opstring,
            couplings;
            sparse=true,
        )
        projected =
            basis.base.symmetry.projector' *
            parent_matrix *
            basis.base.symmetry.projector
        return sparse ? projected : Matrix(projected)
    end
    matrix = _user_operator_csc(
        basis,
        ((opstring, couplings),),
        ComplexF64,
    )
    return sparse ? matrix : Matrix(matrix)
end

function inplace_op!(out, basis::UserBasis, opstring, couplings)
    size(out) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("out must have shape (Ns,Ns)"))
    if _has_symmetry(basis.base.symmetry)
        out .+= operator_matrix(basis, opstring, couplings)
    else
        matrix = _user_operator_csc(
            basis,
            ((opstring, couplings),),
            ComplexF64,
        )
        rows = rowvals(matrix)
        values = nonzeros(matrix)
        for column in axes(matrix, 2)
            for pointer in nzrange(matrix, column)
                out[rows[pointer], column] += values[pointer]
            end
        end
    end
    return out
end

op_bra_ket(basis::UserBasis, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)

function _user_parent_basis(basis::UserBasis)
    _has_symmetry(basis.base.symmetry) || return basis
    parent_base = _discrete_parent_basis(basis.base)
    return UserBasis(
        parent_base,
        basis.basis_dtype,
        basis.op_dict,
        basis.allowed_ops,
        Dict{Symbol,Any}(),
        basis.user_noncommuting_bits,
    )
end

_parent_basis_for_checks(basis::UserBasis) = _user_parent_basis(basis)
check_pcon(basis::UserBasis, static, dynamic=Any[]) =
    check_pcon(basis.base, static, dynamic)

function check_symm(basis::UserBasis, static, dynamic=Any[])
    symmetry = basis.base.symmetry
    _has_symmetry(symmetry) || return true
    parsed = _operator_collections(basis, static, dynamic)
    parsed === nothing && return false
    _, matrices = parsed
    projector = symmetry.projector
    return all(matrices) do matrix
        action = matrix * projector
        residual = action - projector * (projector' * action)
        norm(residual) <= 3e-10 * max(1.0, norm(action))
    end
end

representative(basis::UserBasis, state::Integer) =
    representative(basis.base, state)
normalization(basis::UserBasis, state::Integer) =
    normalization(basis.base, state)
get_amp(basis::UserBasis, state::Integer) =
    get_amp(basis.base, state)

function op_shift_sector(
    target::UserBasis,
    source::UserBasis,
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
)
    target.N == source.N && target.sps == source.sps ||
        throw(ArgumentError(
            "source and target UserBasis objects must have matching local spaces",
        ))
    size(vector, 1) == length(source) ||
        throw(DimensionMismatch(
            "the first vector dimension must equal source Ns",
        ))
    entries = _normalized_shift_entries(operator)
    coefficient_types = Type[eltype(vector)]
    append!(coefficient_types, typeof(entry[3]) for entry in entries)
    T = promote_type(ComplexF64, coefficient_types...)
    parent_input = _parent_coordinates(source.base.symmetry, vector)
    input_matrix = parent_input isa AbstractVector ?
        reshape(parent_input, :, 1) :
        parent_input
    parent_output = zeros(
        T,
        length(target.base.symmetry.parent_states),
        size(input_matrix, 2),
    )
    weights = UInt64[
        UInt64(source.sps)^(site - 1) for site in 1:source.N
    ]
    for (opstring, sites, coupling) in entries
        length(opstring) == length(sites) ||
            throw(ArgumentError("operator arity and sites differ"))
        actions = [
            (_user_operator(source, op), Int(site))
            for (op, site) in Iterators.reverse(
                collect(zip(opstring, sites)),
            )
        ]
        for (column, initial) in
            pairs(source.base.symmetry.parent_states)
            branches = Dict(initial => complex(coupling))
            for (definition, site) in actions
                next_branches = Dict{UInt64,ComplexF64}()
                for (encoded, amplitude) in branches
                    _apply_user_definition!(
                        next_branches,
                        definition,
                        encoded,
                        amplitude,
                        site,
                        source.sps,
                        weights[site],
                    )
                end
                branches = next_branches
            end
            for (encoded, amplitude) in branches
                row = get(
                    target.base.symmetry.parent_lookup,
                    encoded,
                    0,
                )
                row == 0 && continue
                @views parent_output[row, :] .+=
                    amplitude .* input_matrix[column, :]
            end
        end
    end
    result =
        _reduced_coordinates(target.base.symmetry, parent_output)
    vector isa AbstractVector && (result = vec(result))
    out === nothing && return result
    axes(out) == axes(result) ||
        throw(DimensionMismatch(
            "out must have the same axes as the result",
        ))
    copyto!(out, result)
    return out
end

function op_bra_ket(
    basis::UserBasis,
    opstring::AbstractString,
    sites,
    coupling::Number,
    ::Type{T},
    ket_states;
    reduce_output::Bool=true,
) where {T<:Number}
    length(opstring) == length(sites) ||
        throw(ArgumentError("operator arity and sites differ"))
    definitions = [
        (_user_operator(basis, op), Int(site))
        for (op, site) in Iterators.reverse(
            collect(zip(opstring, sites)),
        )
    ]
    all(first(definition) isa Function for definition in definitions) ||
        throw(ArgumentError(
            "UserBasis op_bra_ket requires deterministic callback operators",
        ))
    kets = ket_states isa Integer ?
        UInt64[ket_states] :
        UInt64.(collect(ket_states))
    matrix_elements = zeros(T, length(kets))
    bras = similar(kets)
    for (index, ket) in pairs(kets)
        encoded = ket
        amplitude = complex(coupling)
        for (definition, site) in definitions
            encoded, factor =
                _user_callback_entry(definition, encoded, site)
            amplitude *= factor
        end
        bras[index] = encoded
        matrix_elements[index] = convert(T, amplitude)
    end
    reduce_output || return matrix_elements, bras, kets
    keep = .!iszero.(matrix_elements)
    return matrix_elements[keep], bras[keep], kets[keep]
end

# Deferred general bases deliberately expose no operator behavior until their
# reference states have been constructed. Once materialized, every public
# basis operation delegates to the concrete basis without rebuilding it.
function make_basis!(basis::DeferredBasis)
    _is_materialized(basis) && return basis
    materialized = getfield(basis, :builder)()
    setfield!(basis, :materialized, materialized)
    return basis
end

function make_basis_blocks(basis::DeferredBasis; kwargs...)
    return make_basis_blocks(_materialized(basis); kwargs...)
end

states(basis::DeferredBasis) =
    _is_materialized(basis) ?
    states(_materialized(basis)) :
    copy(getfield(basis, :metadata)[:states])
state_at(basis::DeferredBasis, arguments...; kwargs...) =
    state_at(_materialized(basis), arguments...; kwargs...)
state_index(basis::DeferredBasis, arguments...; kwargs...) =
    state_index(_materialized(basis), arguments...; kwargs...)
int_to_state(basis::DeferredBasis, arguments...; kwargs...) =
    int_to_state(_materialized(basis), arguments...; kwargs...)
state_to_int(basis::DeferredBasis, arguments...; kwargs...) =
    state_to_int(_materialized(basis), arguments...; kwargs...)

projection_matrix(basis::DeferredBasis, arguments...; kwargs...) =
    projection_matrix(_materialized(basis), arguments...; kwargs...)
project_from(basis::DeferredBasis, arguments...; kwargs...) =
    project_from(_materialized(basis), arguments...; kwargs...)
get_vec(basis::DeferredBasis, arguments...; kwargs...) =
    get_vec(_materialized(basis), arguments...; kwargs...)
_full_projection_dimension(basis::DeferredBasis) =
    _full_projection_dimension(_materialized(basis))
_projection_output_dimension(basis::DeferredBasis, pcon::Bool) =
    _projection_output_dimension(_materialized(basis), pcon)
_pcon_projection_matrix(basis::DeferredBasis, ::Type{T}) where {T<:Number} =
    _pcon_projection_matrix(_materialized(basis), T)

operator_matrix(basis::DeferredBasis, arguments...; kwargs...) =
    operator_matrix(_materialized(basis), arguments...; kwargs...)
inplace_op!(output, basis::DeferredBasis, arguments...; kwargs...) =
    inplace_op!(output, _materialized(basis), arguments...; kwargs...)
expanded_form(basis::DeferredBasis, arguments...; kwargs...) =
    expanded_form(_materialized(basis), arguments...; kwargs...)
partial_trace(basis::DeferredBasis, arguments...; kwargs...) =
    partial_trace(_materialized(basis), arguments...; kwargs...)
ent_entropy(basis::DeferredBasis, arguments...; kwargs...) =
    ent_entropy(_materialized(basis), arguments...; kwargs...)

check_hermitian(basis::DeferredBasis, arguments...; kwargs...) =
    check_hermitian(_materialized(basis), arguments...; kwargs...)
check_pcon(basis::DeferredBasis, arguments...; kwargs...) =
    check_pcon(_materialized(basis), arguments...; kwargs...)
check_symm(basis::DeferredBasis, arguments...; kwargs...) =
    check_symm(_materialized(basis), arguments...; kwargs...)
_basis_requires_complex(basis::DeferredBasis) =
    _basis_requires_complex(_materialized(basis))
_parent_basis_for_checks(basis::DeferredBasis) =
    _parent_basis_for_checks(_materialized(basis))

representative(basis::DeferredBasis, state::Integer) =
    representative(_materialized(basis), state)
normalization(basis::DeferredBasis, state::Integer) =
    normalization(_materialized(basis), state)
get_amp(basis::DeferredBasis, state::Integer) =
    get_amp(_materialized(basis), state)
op_bra_ket(basis::DeferredBasis, arguments...; kwargs...) =
    op_bra_ket(_materialized(basis), arguments...; kwargs...)

function op_shift_sector(
    target::DeferredBasis,
    source::DeferredBasis,
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
)
    return op_shift_sector(
        _materialized(target),
        _materialized(source),
        operator,
        vector;
        out,
    )
end

function op_shift_sector(
    target::DeferredBasis,
    source::AbstractBasis,
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
)
    return op_shift_sector(
        _materialized(target),
        source,
        operator,
        vector;
        out,
    )
end

function op_shift_sector(
    target::AbstractBasis,
    source::DeferredBasis,
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
)
    return op_shift_sector(
        target,
        _materialized(source),
        operator,
        vector;
        out,
    )
end
