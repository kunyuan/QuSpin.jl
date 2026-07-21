"""
Basis construction, state encoding, symmetry reduction, projection, and
entanglement utilities used by `QuSpin`.
"""
module Basis

using LinearAlgebra
using SparseArrays

export AbstractBasis, FixedUInt, SpinBasis1D, SpinBasisGeneral
export BosonBasis1D, BosonBasisGeneral
export SpinlessFermionBasis1D, SpinlessFermionBasisGeneral
export SpinfulFermionBasis1D, SpinfulFermionBasisGeneral
export HOBasis, TensorBasis, PhotonBasis, UserBasis
export UInt256, UInt1024, UInt4096, UInt16384
export basis_int_to_python_int, basis_ones, basis_zeros
export bitwise_and, bitwise_leftshift, bitwise_not, bitwise_or
export bitwise_rightshift, bitwise_xor
export coherent_state, get_basis_type, photon_Hspace_dim
export ent_entropy, partial_trace, projection_matrix, python_int_to_basis_int
export expanded_form, get_vec, project_from
export state_at, state_index, states
export int_to_state, state_to_int
export check_hermitian, check_pcon, check_symm, inplace_op!, operator_matrix
export get_amp, make_basis!, make_basis_blocks, normalization
export op_bra_ket, op_shift_sector, project_to, representative
export isbasis

"""
    AbstractBasis

Abstract supertype for every QuSpin basis. Concrete bases define state
enumeration, indexing, projection, and local-operator actions.
"""
abstract type AbstractBasis end

isbasis(value) = value isa AbstractBasis

"""
Internal wrapper used by general bases constructed with `make_basis=false`.
It stores only constructor metadata until `make_basis!` is called.
"""
mutable struct DeferredBasis{
    B<:AbstractBasis,
    F,
    M<:AbstractDict{Symbol,Any},
} <: AbstractBasis
    builder::F
    materialized::Union{Nothing,B}
    metadata::M
end

function _deferred_basis(
    ::Type{B},
    builder::F,
    metadata::M,
) where {
    B<:AbstractBasis,
    F,
    M<:AbstractDict{Symbol,Any},
}
    return DeferredBasis{B,F,M}(builder, nothing, metadata)
end

_is_materialized(basis::DeferredBasis) =
    getfield(basis, :materialized) !== nothing

function _materialized(basis::DeferredBasis)
    result = getfield(basis, :materialized)
    result === nothing &&
        throw(ArgumentError(
            "reference states are not constructed; call make_basis! first",
        ))
    return result
end

function Base.length(basis::DeferredBasis)
    result = getfield(basis, :materialized)
    return result === nothing ? 1 : length(result)
end

function Base.getproperty(basis::DeferredBasis, name::Symbol)
    name in (:builder, :materialized, :metadata) &&
        return getfield(basis, name)
    result = getfield(basis, :materialized)
    if name === :made_basis
        return result !== nothing
    elseif name === :blocks
        if result === nothing
            blocks = copy(getfield(basis, :metadata)[:blocks])
            blocks[:made_basis] = false
            return blocks
        end
        blocks = copy(getproperty(result, :blocks))
        blocks[:made_basis] = true
        return blocks
    elseif result !== nothing
        return getproperty(result, name)
    end
    metadata = getfield(basis, :metadata)
    haskey(metadata, name) && return metadata[name]
    throw(ArgumentError(
        "property '$name' requires a constructed basis; call make_basis! first",
    ))
end

include("symmetry_basis.jl")

"""
    FixedUInt{W}(value)

Unsigned integer with exactly `W` value bits. QuSpin uses 256-, 1024-, 4096-
and 16384-bit basis integers; this Julia-native representation keeps the same
finite-width semantics while storing the payload in a `BigInt`.
"""
struct FixedUInt{W} <: Unsigned
    value::BigInt

    function FixedUInt{W}(value::Integer) where {W}
        W > 0 || throw(ArgumentError("the bit width must be positive"))
        big_value = BigInt(value)
        0 <= big_value < (BigInt(1) << W) ||
            throw(InexactError(:FixedUInt, FixedUInt{W}, value))
        return new{W}(big_value)
    end
end

const UInt256 = FixedUInt{256}
const UInt1024 = FixedUInt{1024}
const UInt4096 = FixedUInt{4096}
const UInt16384 = FixedUInt{16384}

FixedUInt{W}(value::FixedUInt{W}) where {W} = value
Base.BigInt(value::FixedUInt) = copy(value.value)
Base.convert(::Type{BigInt}, value::FixedUInt) = BigInt(value)
Base.convert(::Type{FixedUInt{W}}, value::Integer) where {W} = FixedUInt{W}(value)
Base.convert(::Type{T}, value::FixedUInt) where {T<:Union{UInt8,UInt16,UInt32,UInt64,UInt128}} =
    T(value.value)
Base.convert(::Type{Int}, value::FixedUInt) = Int(value.value)
Base.Int(value::FixedUInt) = Int(value.value)
Base.zero(::Type{FixedUInt{W}}) where {W} = FixedUInt{W}(0)
Base.one(::Type{FixedUInt{W}}) where {W} = FixedUInt{W}(1)
Base.typemin(::Type{FixedUInt{W}}) where {W} = zero(FixedUInt{W})
Base.typemax(::Type{FixedUInt{W}}) where {W} = FixedUInt{W}((BigInt(1) << W) - 1)
Base.iszero(value::FixedUInt) = iszero(value.value)
Base.isone(value::FixedUInt) = isone(value.value)
Base.:(==)(left::FixedUInt, right::FixedUInt) = left.value == right.value
Base.:(==)(left::FixedUInt, right::Integer) = left.value == right
Base.:(==)(left::Integer, right::FixedUInt) = left == right.value
Base.:(==)(left::FixedUInt, right::BigInt) = left.value == right
Base.:(==)(left::BigInt, right::FixedUInt) = left == right.value
Base.isless(left::FixedUInt, right::FixedUInt) = isless(left.value, right.value)
Base.isless(left::FixedUInt, right::Integer) = isless(left.value, right)
Base.isless(left::Integer, right::FixedUInt) = isless(left, right.value)
Base.hash(value::FixedUInt, seed::UInt) = hash(value.value, seed)
Base.show(io::IO, value::FixedUInt{W}) where {W} =
    print(io, "FixedUInt{", W, "}(", value.value, ")")

function _fixed_binary(op, left::FixedUInt{W}, right::Integer) where {W}
    return FixedUInt{W}(op(left.value, BigInt(right)))
end

Base.:&(left::FixedUInt{W}, right::Integer) where {W} = _fixed_binary(&, left, right)
Base.:&(left::Integer, right::FixedUInt{W}) where {W} = right & left
function Base.:&(
    left::FixedUInt{W1},
    right::FixedUInt{W2},
) where {W1,W2}
    W1 == W2 ||
        throw(ArgumentError("FixedUInt bit widths must match"))
    return FixedUInt{W1}(left.value & right.value)
end
Base.:|(left::FixedUInt{W}, right::Integer) where {W} = _fixed_binary(|, left, right)
Base.:|(left::Integer, right::FixedUInt{W}) where {W} = right | left
function Base.:|(
    left::FixedUInt{W1},
    right::FixedUInt{W2},
) where {W1,W2}
    W1 == W2 ||
        throw(ArgumentError("FixedUInt bit widths must match"))
    return FixedUInt{W1}(left.value | right.value)
end
Base.xor(left::FixedUInt{W}, right::Integer) where {W} = _fixed_binary(xor, left, right)
Base.xor(left::Integer, right::FixedUInt{W}) where {W} = xor(right, left)
function Base.xor(
    left::FixedUInt{W1},
    right::FixedUInt{W2},
) where {W1,W2}
    W1 == W2 ||
        throw(ArgumentError("FixedUInt bit widths must match"))
    return FixedUInt{W1}(xor(left.value, right.value))
end
Base.:~(value::FixedUInt{W}) where {W} = FixedUInt{W}(xor(value.value, (BigInt(1) << W) - 1))
Base.:(<<)(value::FixedUInt{W}, shift::Int) where {W} =
    FixedUInt{W}(value.value << shift)
Base.:(>>)(value::FixedUInt{W}, shift::Int) where {W} =
    FixedUInt{W}(value.value >> shift)

"""
    SpinBasis1D(L; nup=nothing, pauli=true)

Computational basis for a spin-one-half chain. Public site indices are
one-based; basis states use the low `L` bits of a `UInt64`.
"""
struct SpinBasis1D <: AbstractBasis
    L::Int
    nup::Union{Nothing,Int}
    pauli::Bool
    encoded_states::Vector{UInt64}
    lookup::Dict{UInt64,Int}
    symmetry::SymmetryData
end

"""
Spin-half general basis whose encoded states exceed 64 bits. It is selected
automatically by `SpinBasisGeneral` and keeps the finite-width integer type in
the basis state vector and lookup table.
"""
struct WideSpinBasis{T<:FixedUInt} <: AbstractBasis
    L::Int
    nup::Int
    pauli::Bool
    encoded_states::Vector{T}
    lookup::Dict{T,Int}
    symmetry::SymmetryData{T}
end

function _fixed_weight_states(L::Int, weight::Int)
    count = binomial(L, weight)
    result = Vector{UInt64}(undef, count)
    if weight == 0
        result[1] = 0
        return result
    elseif weight == L
        result[1] = (UInt64(1) << L) - UInt64(1)
        return result
    end
    state = (UInt64(1) << weight) - UInt64(1)
    for index in eachindex(result)
        result[index] = state
        index == lastindex(result) && break
        lowest = state & (~state + UInt64(1))
        next_prefix = state + lowest
        state =
            next_prefix |
            ((next_prefix ⊻ state) >> (trailing_zeros(lowest) + 2))
    end
    return result
end

function _fixed_weight_states(
    ::Type{T},
    L::Int,
    weight::Int,
) where {T<:FixedUInt}
    count_big = binomial(BigInt(L), BigInt(weight))
    count_big <= typemax(Int) ||
        throw(ArgumentError("the selected particle sector is too large"))
    count = Int(count_big)
    result = Vector{T}(undef, count)
    if weight == 0
        result[1] = zero(T)
        return result
    elseif weight == L
        result[1] = T((BigInt(1) << L) - 1)
        return result
    end
    state = (BigInt(1) << weight) - 1
    for index in eachindex(result)
        result[index] = T(state)
        index == lastindex(result) && break
        lowest = state & -state
        next_prefix = state + lowest
        state =
            next_prefix |
            ((next_prefix ⊻ state) >> (trailing_zeros(lowest) + 2))
    end
    return result
end

function _normalize_general_spin_map(map, L::Int)
    values = Int.(collect(map))
    length(values) == L ||
        throw(ArgumentError("a general symmetry map must contain one entry per site"))
    zero_based = any(iszero, values) || any(<(0), values)
    permutation = Vector{Int}(undef, L)
    flips = falses(L)
    for source in 1:L
        value = values[source]
        if zero_based
            permutation[source] = value >= 0 ? value + 1 : -value
            flips[source] = value < 0
        else
            iszero(value) &&
                throw(ArgumentError("one-based symmetry maps cannot contain zero"))
            permutation[source] = abs(value)
            flips[source] = value < 0
        end
    end
    sort(permutation) == collect(1:L) ||
        throw(ArgumentError("a general symmetry map must be a site permutation"))
    return permutation, flips
end

function _general_spin_map_order(permutation, flips)
    visited = falses(length(permutation))
    order = 1
    for start in eachindex(permutation)
        visited[start] && continue
        current = start
        length_cycle = 0
        flip_parity = false
        while !visited[current]
            visited[current] = true
            length_cycle += 1
            flip_parity ⊻= flips[current]
            current = permutation[current]
        end
        order = lcm(order, flip_parity ? 2length_cycle : length_cycle)
    end
    return order
end

function _general_spin_transform(permutation, flips)
    return state -> begin
        T = typeof(state)
        transformed = zero(T)
        for source in eachindex(permutation)
            occupied = !iszero(state & (one(T) << (source - 1)))
            flips[source] && (occupied = !occupied)
            occupied || continue
            transformed |= one(T) << (permutation[source] - 1)
        end
        return transformed, 1.0 + 0im
    end
end

function _parse_spin_value(S)
    if S isa Rational
        value = S
    elseif S isa Integer
        value = S // 1
    elseif S isa AbstractString
        pieces = split(strip(S), "/")
        value = length(pieces) == 1 ?
            parse(Int, pieces[1]) // 1 :
            length(pieces) == 2 ?
                parse(Int, pieces[1]) // parse(Int, pieces[2]) :
                throw(ArgumentError("S must be an integer or half-integer"))
    elseif S isa Real
        value = rationalize(S; tol=eps(float(S)) * 8)
    else
        throw(ArgumentError("S must be an integer or half-integer"))
    end
    value > 0 && denominator(value) in (1, 2) ||
        throw(ArgumentError("S must be a positive integer or half-integer"))
    return value
end

function _wide_spin_general_basis(
    L::Int,
    wanted::Int,
    pauli::Bool,
    block_order,
    blocks,
)
    T = get_basis_type(L, wanted, 2)
    T <: FixedUInt ||
        throw(ArgumentError("wide spin bases require a fixed-width integer type"))
    parent_states = _fixed_weight_states(T, L, wanted)
    lookup =
        Dict(state => index for (index, state) in pairs(parent_states))
    symmetry_blocks =
        Dict{Symbol,Any}(:nup => wanted, :pauli => pauli)
    isempty(blocks) && return WideSpinBasis(
        L,
        wanted,
        pauli,
        parent_states,
        lookup,
        _identity_symmetry_data(
            parent_states,
            symmetry_blocks,
            lookup,
        ),
    )

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

    projector = nothing
    for name in ordered_names
        specification = block_dictionary[name]
        (specification isa Tuple || specification isa AbstractVector) &&
            length(specification) >= 2 ||
            throw(ArgumentError(
                "general symmetry blocks must be (map, quantum_number)",
            ))
        map, quantum_number = specification[1], Int(specification[2])
        permutation, flips = _normalize_general_spin_map(map, L)
        order = _general_spin_map_order(permutation, flips)
        eigenvalue = cis(2π * mod(quantum_number, order) / order)
        transform = _general_spin_transform(permutation, flips)
        projector = projector === nothing ?
            _cyclic_projector(
                parent_states,
                lookup,
                transform,
                ComplexF64(eigenvalue),
            ) :
            _intersect_eigenspace(
                projector,
                _signed_permutation(parent_states, lookup, transform),
                ComplexF64(eigenvalue),
            )
        symmetry_blocks[name] = mod(quantum_number, order)
        symmetry_blocks[Symbol(name, :_period)] = order
    end
    symmetry_blocks[:block_order] = copy(ordered_names)
    symmetry = _finalize_symmetry_data(
        parent_states,
        projector,
        symmetry_blocks,
    )
    representatives =
        _representative_states(parent_states, symmetry.projector)
    return WideSpinBasis(
        L,
        wanted,
        pauli,
        representatives,
        Dict(state => index for (index, state) in pairs(representatives)),
        symmetry,
    )
end

"""
    SpinBasisGeneral(N; Nup=nothing, pauli=true, block_order=nothing, blocks...)

Spin-half basis reduced by arbitrary finite-order site maps. Each symmetry
block is supplied as `name=(map, quantum_number)`. Maps may use Julia's
one-based site labels or QuSpin's zero-based labels; negative QuSpin labels
encode a spin inversion at the mapped site.
"""
function SpinBasisGeneral(
    L::Integer;
    nup::Union{Nothing,Integer}=nothing,
    Nup=nothing,
    m=nothing,
    S="1/2",
    pauli::Bool=true,
    block_order=nothing,
    make_basis::Bool=true,
    Ns_block_est=nothing,
    blocks...,
)
    if !make_basis
        block_keywords = (; blocks...)
        spin_value = _parse_spin_value(S)
        selected_nup = nup === nothing ? Nup : nup
        if selected_nup === nothing && m !== nothing
            selected_nup = round(Int, (float(m) + 0.5) * L)
        end
        target_type = if spin_value != 1 // 2
            DiscreteBasis{:spin}
        elseif L > 63
            WideSpinBasis{get_basis_type(L, selected_nup, 2)}
        else
            SpinBasis1D
        end
        builder = () -> SpinBasisGeneral(
            L;
            nup,
            Nup,
            m,
            S,
            pauli,
            block_order,
            make_basis=true,
            Ns_block_est,
            block_keywords...,
        )
        metadata = Dict{Symbol,Any}(
            :L => Int(L),
            :N => Int(L),
            :Ns => 1,
            :sps => Int(2spin_value + 1),
            :dtype => L > 63 ?
                get_basis_type(L, selected_nup, Int(2spin_value + 1)) :
                UInt64,
            :states => L > 63 ?
                [zero(get_basis_type(L, selected_nup, Int(2spin_value + 1)))] :
                UInt64[0],
            :encoded_states => L > 63 ?
                [zero(get_basis_type(L, selected_nup, Int(2spin_value + 1)))] :
                UInt64[0],
            :nup => selected_nup,
            :pauli => pauli,
            :description => spin_value == 1 // 2 ?
                "deferred spin-1/2 general basis" :
                "deferred higher-spin general basis",
            :operators => ("I", "z", "+", "-", "x", "y"),
            :noncommuting_bits => Tuple{Vector{Int},Int}[],
            :blocks => Dict{Symbol,Any}(
                Symbol(name) => value for (name, value) in blocks
            ),
        )
        Ns_block_est === nothing ||
            (metadata[:Ns_block_est] = Int(Ns_block_est))
        return _deferred_basis(target_type, builder, metadata)
    end
    spin_value = _parse_spin_value(S)
    if spin_value == 1 // 2 && L > 63
        nup !== nothing && Nup !== nothing &&
            throw(ArgumentError("specify only one of nup and Nup"))
        selected_nup = nup === nothing ? Nup : nup
        if m !== nothing
            selected_nup === nothing ||
                throw(ArgumentError("m cannot be combined with nup or Nup"))
            selected_nup = round(Int, (float(m) + 0.5) * L)
        end
        selected_nup === nothing &&
            throw(ArgumentError(
                "wide spin bases require nup, Nup, or m to avoid enumerating 2^L states",
            ))
        0 <= selected_nup <= L ||
            throw(ArgumentError("nup must lie between 0 and L"))
        return _wide_spin_general_basis(
            Int(L),
            Int(selected_nup),
            pauli,
            block_order,
            blocks,
        )
    end
    base = SpinBasis1D(
        L;
        nup,
        Nup,
        m,
        S,
        pauli,
    )
    if base isa DiscreteBasis
        return _general_discrete_basis(base, block_order, blocks)
    end
    isempty(blocks) && return base
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

    parent_states = base.encoded_states
    lookup = base.lookup
    projector = nothing
    symmetry_blocks = copy(base.symmetry.blocks)
    for name in ordered_names
        specification = block_dictionary[name]
        (specification isa Tuple || specification isa AbstractVector) &&
            length(specification) >= 2 ||
            throw(ArgumentError(
                "general symmetry blocks must be (map, quantum_number)",
            ))
        map, quantum_number = specification[1], Int(specification[2])
        permutation, flips =
            _normalize_general_spin_map(map, Int(L))
        order = _general_spin_map_order(permutation, flips)
        eigenvalue = cis(2π * mod(quantum_number, order) / order)
        transform = _general_spin_transform(permutation, flips)
        projector = if projector === nothing
            _cyclic_projector(
                parent_states,
                lookup,
                transform,
                ComplexF64(eigenvalue),
            )
        else
            _intersect_eigenspace(
                projector,
                _signed_permutation(parent_states, lookup, transform),
                ComplexF64(eigenvalue),
            )
        end
        symmetry_blocks[name] = mod(quantum_number, order)
        symmetry_blocks[Symbol(name, :_period)] = order
    end
    symmetry_blocks[:block_order] = copy(ordered_names)
    symmetry = _finalize_symmetry_data(
        parent_states,
        projector,
        symmetry_blocks,
    )
    representatives =
        _representative_states(parent_states, symmetry.projector)
    return SpinBasis1D(
        Int(L),
        base.nup,
        pauli,
        representatives,
        Dict(state => index for (index, state) in pairs(representatives)),
        symmetry,
    )
end

function SpinBasis1D(
    L::Integer;
    nup::Union{Nothing,Integer}=nothing,
    Nup=nothing,
    m=nothing,
    S="1/2",
    pauli::Bool=true,
    a::Integer=1,
    kblock=nothing,
    pblock=nothing,
    zblock=nothing,
    pzblock=nothing,
    zAblock=nothing,
    zBblock=nothing,
)
    spin_value = _parse_spin_value(S)
    if spin_value != 1 // 2
        return DiscreteBasis{:spin}(
            L;
            S=spin_value,
            nup,
            Nup,
            m,
            pauli,
            a,
            kblock,
            pblock,
            zblock,
            pzblock,
            zAblock,
            zBblock,
        )
    end
    1 <= L <= 63 || throw(ArgumentError("L must be between 1 and 63"))
    nup !== nothing && Nup !== nothing &&
        throw(ArgumentError("specify only one of nup and Nup"))
    selected_nup = nup === nothing ? Nup : nup
    if m !== nothing
        selected_nup === nothing ||
            throw(ArgumentError("m cannot be combined with nup or Nup"))
        selected_nup = round(Int, (float(m) + 0.5) * L)
    end
    selected_nup === nothing || 0 <= selected_nup <= L ||
        throw(ArgumentError("nup must lie between 0 and L"))

    wanted = selected_nup === nothing ? nothing : Int(selected_nup)
    encoded = wanted === nothing ?
        collect(UInt64(0):((UInt64(1) << Int(L)) - UInt64(1))) :
        _fixed_weight_states(Int(L), wanted)
    lookup = Dict(state => i for (i, state) in pairs(encoded))
    blocks = Dict{Symbol,Any}(:nup => wanted, :pauli => pauli)
    requested = (
        kblock !== nothing ||
        pblock !== nothing ||
        zblock !== nothing ||
        pzblock !== nothing ||
        zAblock !== nothing ||
        zBblock !== nothing
    )
    if !requested
        symmetry = _identity_symmetry_data(encoded, blocks, lookup)
        return SpinBasis1D(Int(L), wanted, pauli, encoded, lookup, symmetry)
    end

    a > 0 && L % a == 0 ||
        throw(ArgumentError("a must be a positive divisor of L"))
    order = Int(L ÷ a)
    projector = nothing

    function site_permutation_transform(permutation)
        return state -> begin
            transformed = zero(UInt64)
            for site in 1:Int(L)
                iszero(state & (UInt64(1) << (site - 1))) && continue
                transformed |= UInt64(1) << (permutation[site] - 1)
            end
            transformed, 1.0 + 0im
        end
    end
    translation = [mod1(site + Int(a), Int(L)) for site in 1:Int(L)]
    parity = [Int(L) - site + 1 for site in 1:Int(L)]

    if kblock !== nothing
        momentum = mod(Int(kblock), order)
        eigenvalue = cis(2π * momentum / order)
        projector = _cyclic_projector(
            encoded,
            lookup,
            site_permutation_transform(translation),
            eigenvalue,
        )
        blocks[:a] = Int(a)
        blocks[:kblock] = momentum
    end

    if pblock !== nothing && kblock !== nothing
        momentum = mod(Int(kblock), order)
        (momentum == 0 || (iseven(order) && momentum == order ÷ 2)) ||
            throw(ArgumentError(
            "pblock can be combined with kblock only at k=0 or pi",
        ))
    end

    flip_mask = (UInt64(1) << Int(L)) - UInt64(1)
    flip_transform = state -> (xor(state, flip_mask), 1.0 + 0im)
    parity_transform = site_permutation_transform(parity)
    compose(first_transform, second_transform) = state -> begin
        intermediate, first_phase = first_transform(state)
        transformed, second_phase = second_transform(intermediate)
        transformed, first_phase * second_phase
    end
    odd_mask = sum(UInt64(1) << (site - 1) for site in 1:2:Int(L))
    even_mask = sum(UInt64(1) << (site - 1) for site in 2:2:Int(L))

    for (name, value, transform) in (
        (:pblock, pblock, parity_transform),
        (:zblock, zblock, flip_transform),
        (:pzblock, pzblock, compose(parity_transform, flip_transform)),
        (:zAblock, zAblock, state -> (xor(state, odd_mask), 1.0 + 0im)),
        (:zBblock, zBblock, state -> (xor(state, even_mask), 1.0 + 0im)),
    )
        value === nothing && continue
        value in (-1, 1) ||
            throw(ArgumentError("$name must be +1 or -1"))
        projector = if projector === nothing
            _cyclic_projector(
                encoded,
                lookup,
                transform,
                ComplexF64(value),
            )
        else
            symmetry_matrix =
                _signed_permutation(encoded, lookup, transform)
            _intersect_eigenspace(
                projector,
                symmetry_matrix,
                ComplexF64(value),
            )
        end
        blocks[name] = Int(value)
    end

    projector === nothing &&
        throw(ArgumentError("at least one symmetry block must be specified"))
    symmetry = _finalize_symmetry_data(encoded, projector, blocks)
    representatives = _representative_states(encoded, symmetry.projector)
    return SpinBasis1D(
        Int(L),
        wanted,
        pauli,
        representatives,
        Dict(state => index for (index, state) in pairs(representatives)),
        symmetry,
    )
end

Base.length(basis::SpinBasis1D) = length(basis.encoded_states)
Base.:(==)(left::SpinBasis1D, right::SpinBasis1D) =
    left.L == right.L &&
    left.nup == right.nup &&
    left.pauli == right.pauli &&
    left.encoded_states == right.encoded_states &&
    left.symmetry.blocks == right.symmetry.blocks
states(basis::SpinBasis1D) = copy(basis.encoded_states)

function Base.getproperty(basis::SpinBasis1D, name::Symbol)
    name === :N && return getfield(basis, :L)
    name === :Ns && return length(getfield(basis, :encoded_states))
    name === :blocks && return copy(getfield(basis, :symmetry).blocks)
    name === :description && return "spin-1/2 chain basis"
    name === :dtype && return UInt64
    name === :noncommuting_bits && return Tuple{Vector{Int},Int}[]
    name === :operators && return ("I", "z", "+", "-", "x", "y")
    name === :sps && return 2
    name === :states && return copy(getfield(basis, :encoded_states))
    return getfield(basis, name)
end

Base.length(basis::WideSpinBasis) = length(basis.encoded_states)
states(basis::WideSpinBasis) = copy(basis.encoded_states)

function Base.getproperty(basis::WideSpinBasis{T}, name::Symbol) where {T}
    name === :N && return getfield(basis, :L)
    name === :Ns && return length(getfield(basis, :encoded_states))
    name === :blocks && return copy(getfield(basis, :symmetry).blocks)
    name === :description && return "wide-integer spin-1/2 general basis"
    name === :dtype && return T
    name === :noncommuting_bits && return Tuple{Vector{Int},Int}[]
    name === :operators && return ("I", "z", "+", "-", "x", "y")
    name === :sps && return 2
    name === :states && return copy(getfield(basis, :encoded_states))
    return getfield(basis, name)
end

function projection_matrix(
    basis::WideSpinBasis,
    ::Type{T}=Float64;
    sparse::Bool=false,
    pcon::Bool=false,
) where {T<:Number}
    if pcon
        projected = _parent_projection_matrix(basis.symmetry, T)
        return sparse ? projected : Matrix(projected)
    end
    throw(ArgumentError(
        "a 2^$(basis.L)-row full-space projector cannot be represented by Julia array dimensions; operate in the reduced basis instead",
    ))
end

_full_projection_dimension(basis::WideSpinBasis) =
    throw(ArgumentError(
        "the full Hilbert-space dimension exceeds Julia array dimensions",
    ))
_projection_output_dimension(basis::WideSpinBasis, pcon::Bool) =
    pcon ? length(basis.symmetry.parent_states) :
    _full_projection_dimension(basis)
_pcon_projection_matrix(basis::WideSpinBasis, ::Type{T}) where {T<:Number} =
    _parent_projection_matrix(basis.symmetry, T)
function project_from(
    basis::WideSpinBasis,
    vector::AbstractVecOrMat;
    sparse::Bool=true,
    pcon::Bool=false,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    pcon &&
        return _project_from_parent(
            basis.symmetry,
            vector,
            sparse,
        )
    throw(ArgumentError(
        "wide bases cannot materialize a vector in the full 2^L Hilbert space",
    ))
end
get_vec(basis::WideSpinBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)

function projection_matrix(
    basis::SpinBasis1D,
    ::Type{T}=Float64,
    ;
    sparse::Bool=false,
    pcon::Bool=false,
) where {T<:Number}
    if pcon
        projected = _parent_projection_matrix(basis.symmetry, T)
        return sparse ? projected : Matrix(projected)
    end
    return _full_projection_matrix(
        basis.encoded_states,
        basis.symmetry,
        1 << basis.L,
        T,
        sparse_output=sparse,
    )
end

_full_projection_dimension(basis::SpinBasis1D) = 1 << basis.L
_projection_output_dimension(basis::SpinBasis1D, pcon::Bool) =
    pcon ? length(basis.symmetry.parent_states) :
    _full_projection_dimension(basis)
_pcon_projection_matrix(basis::SpinBasis1D, ::Type{T}) where {T<:Number} =
    _parent_projection_matrix(basis.symmetry, T)

function project_from(
    basis::SpinBasis1D,
    vector::AbstractVecOrMat;
    sparse::Bool=true,
    pcon::Bool=false,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    pcon &&
        return _project_from_parent(
            basis.symmetry,
            vector,
            sparse,
        )
    return _project_from_full(
        basis.symmetry,
        vector,
        1 << basis.L,
        sparse,
    )
end

get_vec(basis::SpinBasis1D, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)

function _xy_expansion_choices(::SpinBasis1D, operator::Char)
    operator == 'x' &&
        return (('+', 0.5 + 0im), ('-', 0.5 + 0im))
    operator == 'y' &&
        return (('+', 0.5im), ('-', -0.5im))
    return ((operator, 1.0 + 0im),)
end

function _xy_expansion_choices(::WideSpinBasis, operator::Char)
    operator == 'x' &&
        return (('+', 0.5 + 0im), ('-', 0.5 + 0im))
    operator == 'y' &&
        return (('+', 0.5im), ('-', -0.5im))
    return ((operator, 1.0 + 0im),)
end

function _expanded_operator_strings(basis, opstring::AbstractString)
    expansions = Tuple{String,ComplexF64}[("", 1.0 + 0im)]
    for operator in opstring
        choices = operator == '|' ?
            (('|', 1.0 + 0im),) :
            _xy_expansion_choices(basis, operator)
        expansions = [
            (
                prefix * string(replacement),
                coefficient * scale,
            )
            for (prefix, coefficient) in expansions
            for (replacement, scale) in choices
        ]
    end
    return expansions
end

function _scaled_couplings(couplings, scale)
    return [
        (scale * first(coupling), Base.tail(Tuple(coupling))...)
        for coupling in couplings
    ]
end

function _expanded_form_entries(basis, entries, dynamic::Bool)
    expanded = Any[]
    for entry in entries
        if hasproperty(entry, :op) && hasproperty(entry, :couplings)
            dynamic && throw(ArgumentError(
                "dynamic OperatorTerm entries require a drive and arguments",
            ))
            for (opstring, scale) in
                _expanded_operator_strings(basis, getproperty(entry, :op))
                push!(
                    expanded,
                    Any[
                        opstring,
                        _scaled_couplings(
                            getproperty(entry, :couplings),
                            scale,
                        ),
                    ],
                )
            end
            continue
        end
        if !(entry isa Tuple || entry isa AbstractVector) ||
           isempty(entry) ||
           !(first(entry) isa AbstractString)
            push!(expanded, entry)
            continue
        end
        expected_length = dynamic ? 4 : 2
        length(entry) == expected_length ||
            throw(ArgumentError(
                dynamic ?
                "dynamic entries are [op, couplings, f, f_args]" :
                "static entries are [op, couplings]",
            ))
        opstring, couplings = entry[1], entry[2]
        for (expanded_op, scale) in
            _expanded_operator_strings(basis, opstring)
            scaled = _scaled_couplings(couplings, scale)
            if dynamic
                push!(
                    expanded,
                    Any[expanded_op, scaled, entry[3], entry[4]],
                )
            else
                push!(expanded, Any[expanded_op, scaled])
            end
        end
    end
    return expanded
end

expanded_form(basis::SpinBasis1D, static=Any[], dynamic=Any[]) = (
    _expanded_form_entries(basis, static, false),
    _expanded_form_entries(basis, dynamic, true),
)
expanded_form(basis::WideSpinBasis, static=Any[], dynamic=Any[]) = (
    _expanded_form_entries(basis, static, false),
    _expanded_form_entries(basis, dynamic, true),
)

function state_index(basis::SpinBasis1D, state::Integer)
    state < 0 && throw(ArgumentError("state must be nonnegative"))
    encoded = UInt64(state)
    return get(basis.lookup, encoded) do
        throw(ArgumentError("state $state is not represented by this basis"))
    end
end

function state_at(basis::SpinBasis1D, index::Integer)
    checkbounds(basis.encoded_states, index)
    return basis.encoded_states[index]
end

function int_to_state(
    basis::SpinBasis1D,
    state::Integer;
    bracket_notation::Bool=true,
)
    0 <= state < (BigInt(1) << basis.L) ||
        throw(ArgumentError("state must fit in L bits"))
    bits = string(state; base=2, pad=basis.L)
    return bracket_notation ? "|" * join(bits, " ") * ">" : bits
end

function state_to_int(basis::SpinBasis1D, state::AbstractString)
    compact = replace(strip(state, ['|', '>']), " " => "")
    length(compact) == basis.L ||
        throw(ArgumentError("state must contain exactly $(basis.L) binary digits"))
    all(bit -> bit in ('0', '1'), compact) ||
        throw(ArgumentError("state must be binary"))
    return parse(UInt64, compact; base=2)
end

function state_index(basis::WideSpinBasis{T}, state::Integer) where {T}
    state < 0 && throw(ArgumentError("state must be nonnegative"))
    encoded = T(state)
    return get(basis.lookup, encoded) do
        throw(ArgumentError("state $state is not represented by this basis"))
    end
end

function state_at(basis::WideSpinBasis, index::Integer)
    checkbounds(basis.encoded_states, index)
    return basis.encoded_states[index]
end

function int_to_state(
    basis::WideSpinBasis,
    state::Integer;
    bracket_notation::Bool=true,
)
    value = BigInt(state)
    0 <= value < (BigInt(1) << basis.L) ||
        throw(ArgumentError("state must fit in L bits"))
    bits = string(value; base=2, pad=basis.L)
    return bracket_notation ? "|" * join(bits, " ") * ">" : bits
end

function state_to_int(
    basis::WideSpinBasis{T},
    state::AbstractString,
) where {T}
    compact = replace(strip(state, ['|', '>']), " " => "")
    length(compact) == basis.L ||
        throw(ArgumentError(
            "state must contain exactly $(basis.L) binary digits",
        ))
    all(bit -> bit in ('0', '1'), compact) ||
        throw(ArgumentError("state must be binary"))
    return T(parse(BigInt, compact; base=2))
end

function _apply_spin_local(
    basis::SpinBasis1D,
    state::UInt64,
    op::Char,
    site::Integer,
)
    1 <= site <= basis.L ||
        throw(ArgumentError("site must lie in 1:$(basis.L)"))
    mask = UInt64(1) << (site - 1)
    occupied = !iszero(state & mask)
    if op == 'I'
        return state, 1.0, true
    elseif op == 'z'
        scale = basis.pauli ? 1.0 : 0.5
        return state, occupied ? scale : -scale, true
    elseif op == '+'
        occupied && return state, 0.0, false
        return state | mask, basis.pauli ? 2.0 : 1.0, true
    elseif op == '-'
        occupied || return state, 0.0, false
        return state & ~mask, basis.pauli ? 2.0 : 1.0, true
    elseif op == 'x'
        return xor(state, mask), basis.pauli ? 1.0 : 0.5, true
    elseif op == 'y'
        scale = basis.pauli ? 1.0 : 0.5
        return xor(state, mask), occupied ? -im * scale : im * scale, true
    end
    throw(ArgumentError("unsupported spin operator '$op'"))
end

function _spin_operator_triplets(
    basis::SpinBasis1D,
    opstring::AbstractString,
    couplings,
)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    for coupling in couplings
        length(coupling) == length(opstring) + 1 ||
            throw(ArgumentError("operator arity and sites differ"))
        sites = Base.tail(coupling)
        for (column, initial) in pairs(basis.encoded_states)
            state = initial
            amplitude = complex(first(coupling))
            alive = true
            for index in length(opstring):-1:1
                state, factor, alive = _apply_spin_local(
                    basis,
                    state,
                    opstring[index],
                    sites[index],
                )
                alive || break
                amplitude *= factor
            end
            alive || continue
            row = get(basis.lookup, state, 0)
            row == 0 && continue
            push!(rows, row)
            push!(columns, column)
            push!(values, amplitude)
        end
    end
    return rows, columns, values
end

function _accumulate_triplets!(out, rows, columns, values)
    for index in eachindex(values)
        out[rows[index], columns[index]] += values[index]
    end
    return out
end

function operator_matrix(
    basis::SpinBasis1D,
    opstring::AbstractString,
    couplings,
    ;
    sparse::Bool=false,
)
    if _has_symmetry(basis.symmetry)
        parent = _spin_parent_basis(basis)
        rows, columns, values =
            _spin_operator_triplets(parent, opstring, couplings)
        projected = _projected_triplet_matrix(
            basis.symmetry.projector,
            rows,
            columns,
            values,
        )
        return sparse ? projected : Matrix(projected)
    end
    rows, columns, values =
        _spin_operator_triplets(basis, opstring, couplings)
    matrix = SparseArrays.sparse(
        rows,
        columns,
        values,
        length(basis),
        length(basis),
    )
    return sparse ? matrix : Matrix(matrix)
end

function inplace_op!(out, basis::SpinBasis1D, opstring, couplings)
    size(out) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("out must have shape (Ns,Ns)"))
    if _has_symmetry(basis.symmetry)
        parent = _spin_parent_basis(basis)
        rows, columns, values =
            _spin_operator_triplets(parent, opstring, couplings)
        return _accumulate_projected_triplets!(
            out,
            basis.symmetry.projector,
            rows,
            columns,
            values,
        )
    end
    rows, columns, values =
        _spin_operator_triplets(basis, opstring, couplings)
    _accumulate_triplets!(out, rows, columns, values)
    return out
end

op_bra_ket(basis::SpinBasis1D, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)

function _wide_spin_parent_basis(basis::WideSpinBasis{T}) where {T}
    parent_states = basis.symmetry.parent_states
    parent_lookup = basis.symmetry.parent_lookup
    blocks = Dict{Symbol,Any}(:nup => basis.nup, :pauli => basis.pauli)
    return WideSpinBasis(
        basis.L,
        basis.nup,
        basis.pauli,
        parent_states,
        parent_lookup,
        _identity_symmetry_data(parent_states, blocks, parent_lookup),
    )
end

function _apply_wide_spin_local(
    basis::WideSpinBasis{T},
    state::T,
    op::Char,
    site::Integer,
) where {T}
    1 <= site <= basis.L ||
        throw(ArgumentError("site must lie in 1:$(basis.L)"))
    mask = one(T) << (site - 1)
    occupied = !iszero(state & mask)
    if op == 'I'
        return state, 1.0, true
    elseif op == 'z'
        scale = basis.pauli ? 1.0 : 0.5
        return state, occupied ? scale : -scale, true
    elseif op == '+'
        occupied && return state, 0.0, false
        return state | mask, basis.pauli ? 2.0 : 1.0, true
    elseif op == '-'
        occupied || return state, 0.0, false
        return state & ~mask, basis.pauli ? 2.0 : 1.0, true
    elseif op == 'x'
        return xor(state, mask), basis.pauli ? 1.0 : 0.5, true
    elseif op == 'y'
        scale = basis.pauli ? 1.0 : 0.5
        return xor(state, mask), occupied ? -im * scale : im * scale, true
    end
    throw(ArgumentError("unsupported spin operator '$op'"))
end

function _wide_spin_operator_triplets(
    basis::WideSpinBasis,
    opstring::AbstractString,
    couplings,
)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    for coupling in couplings
        length(coupling) == length(opstring) + 1 ||
            throw(ArgumentError("operator arity and sites differ"))
        sites = Base.tail(coupling)
        for (column, initial) in pairs(basis.encoded_states)
            state = initial
            amplitude = complex(first(coupling))
            alive = true
            for index in length(opstring):-1:1
                state, factor, alive = _apply_wide_spin_local(
                    basis,
                    state,
                    opstring[index],
                    sites[index],
                )
                alive || break
                amplitude *= factor
            end
            alive || continue
            row = get(basis.lookup, state, 0)
            row == 0 && continue
            push!(rows, row)
            push!(columns, column)
            push!(values, amplitude)
        end
    end
    return rows, columns, values
end

function operator_matrix(
    basis::WideSpinBasis,
    opstring::AbstractString,
    couplings;
    sparse::Bool=false,
)
    parent = _has_symmetry(basis.symmetry) ?
        _wide_spin_parent_basis(basis) :
        basis
    rows, columns, values =
        _wide_spin_operator_triplets(parent, opstring, couplings)
    matrix = if _has_symmetry(basis.symmetry)
        _projected_triplet_matrix(
            basis.symmetry.projector,
            rows,
            columns,
            values,
        )
    else
        SparseArrays.sparse(
            rows,
            columns,
            values,
            length(basis),
            length(basis),
        )
    end
    return sparse ? matrix : Matrix(matrix)
end

function inplace_op!(output, basis::WideSpinBasis, opstring, couplings)
    output .+= operator_matrix(basis, opstring, couplings)
    return output
end

op_bra_ket(basis::WideSpinBasis, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)

function op_bra_ket(
    basis::WideSpinBasis{S},
    opstring::AbstractString,
    sites,
    coupling::Number,
    ::Type{T},
    ket_states;
    reduce_output::Bool=true,
) where {S,T<:Number}
    length(opstring) == length(sites) ||
        throw(ArgumentError("operator arity and sites differ"))
    kets = ket_states isa Integer ?
        S[ket_states] :
        S.(collect(ket_states))
    matrix_elements = zeros(T, length(kets))
    bras = similar(kets)
    for (position, ket) in pairs(kets)
        state = ket
        amplitude = complex(coupling)
        alive = true
        for index in length(opstring):-1:1
            state, factor, alive = _apply_wide_spin_local(
                basis,
                state,
                opstring[index],
                Int(sites[index]),
            )
            alive || break
            amplitude *= factor
        end
        bras[position] = state
        if alive
            matrix_elements[position] = convert(T, amplitude)
        end
    end
    reduce_output || return matrix_elements, bras, kets
    keep = .!iszero.(matrix_elements)
    return matrix_elements[keep], bras[keep], kets[keep]
end

function _subsystem_sites(basis::SpinBasis1D, sub_sys_A)
    sites = sub_sys_A === nothing ?
        collect(1:fld(basis.L, 2)) :
        Int.(collect(sub_sys_A))
    allunique(sites) || throw(ArgumentError("subsystem sites must be unique"))
    all(site -> 1 <= site <= basis.L, sites) ||
        throw(ArgumentError("subsystem sites must lie in 1:$(basis.L)"))
    return sites
end

function _subsystem_index(state::UInt64, sites)
    index = 0
    for site in sites
        index = (index << 1) | Int((state >> (site - 1)) & 1)
    end
    return index + 1
end

function _reduced_density_matrices(
    basis::SpinBasis1D,
    state::AbstractVector,
    sites_A,
    return_rdm=:both,
)
    length(state) == length(basis) ||
        throw(DimensionMismatch("state length must equal the basis dimension"))
    sites_B = setdiff(collect(1:basis.L), sites_A)
    coefficients = zeros(
        eltype(state),
        1 << length(sites_A),
        1 << length(sites_B),
    )
    for (position, encoded) in pairs(basis.encoded_states)
        row = _subsystem_index(encoded, sites_A)
        column = _subsystem_index(encoded, sites_B)
        coefficients[row, column] = state[position]
    end
    if return_rdm in (:A, "A")
        return coefficients * coefficients'
    elseif return_rdm in (:B, "B")
        return coefficients' * coefficients
    elseif return_rdm in (:both, "both")
        return coefficients * coefficients', coefficients' * coefficients
    end
    throw(ArgumentError("return_rdm must be A, B, or both"))
end

function _spin_parent_basis(basis::SpinBasis1D)
    parent_states = basis.symmetry.parent_states
    parent_lookup = basis.symmetry.parent_lookup
    blocks = Dict{Symbol,Any}(:nup => basis.nup, :pauli => basis.pauli)
    return SpinBasis1D(
        basis.L,
        basis.nup,
        basis.pauli,
        parent_states,
        parent_lookup,
        _identity_symmetry_data(parent_states, blocks, parent_lookup),
    )
end

function _spin_pure_coefficients(
    basis::SpinBasis1D,
    state::AbstractVector,
    sites_A,
)
    parent_basis = basis
    parent_state = state
    if _has_symmetry(basis.symmetry)
        parent_basis = _spin_parent_basis(basis)
        parent_state = basis.symmetry.projector * state
    end
    length(parent_state) == length(parent_basis) ||
        throw(DimensionMismatch("state length must equal the basis dimension"))
    sites_B = setdiff(collect(1:parent_basis.L), sites_A)
    coefficients = zeros(
        eltype(parent_state),
        1 << length(sites_A),
        1 << length(sites_B),
    )
    for (position, encoded) in pairs(parent_basis.encoded_states)
        row = _subsystem_index(encoded, sites_A)
        column = _subsystem_index(encoded, sites_B)
        coefficients[row, column] = parent_state[position]
    end
    return coefficients
end

function _reduced_density_matrices(
    basis::SpinBasis1D,
    state::AbstractMatrix,
    sites_A,
    return_rdm=:both,
)
    size(state) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("density matrix must match the basis dimension"))
    sites_B = setdiff(collect(1:basis.L), sites_A)
    need_A = return_rdm in (:A, "A", :both, "both")
    need_B = return_rdm in (:B, "B", :both, "both")
    need_A || need_B ||
        throw(ArgumentError("return_rdm must be A, B, or both"))
    rho_A = need_A ?
        zeros(eltype(state), 1 << length(sites_A), 1 << length(sites_A)) :
        nothing
    rho_B = need_B ?
        zeros(eltype(state), 1 << length(sites_B), 1 << length(sites_B)) :
        nothing
    indices_A = Int[
        _subsystem_index(encoded, sites_A)
        for encoded in basis.encoded_states
    ]
    indices_B = Int[
        _subsystem_index(encoded, sites_B)
        for encoded in basis.encoded_states
    ]
    if need_A
        groups_B = Dict{Int,Vector{Int}}()
        for position in eachindex(indices_B)
            push!(get!(Vector{Int}, groups_B, indices_B[position]), position)
        end
        for group in values(groups_B),
            row_position in group,
            column_position in group
            rho_A[
                indices_A[row_position],
                indices_A[column_position],
            ] += state[row_position, column_position]
        end
    end
    if need_B
        groups_A = Dict{Int,Vector{Int}}()
        for position in eachindex(indices_A)
            push!(get!(Vector{Int}, groups_A, indices_A[position]), position)
        end
        for group in values(groups_A),
            row_position in group,
            column_position in group
            rho_B[
                indices_B[row_position],
                indices_B[column_position],
            ] += state[row_position, column_position]
        end
    end
    return need_A && need_B ? (rho_A, rho_B) :
           need_A ? rho_A : rho_B
end

"""
    partial_trace(basis, state; sub_sys_A=nothing, return_rdm=:A)

Reduced density matrix for a pure state or density matrix represented in a
`SpinBasis1D`, including particle-number-restricted bases.
"""
function partial_trace(
    basis::SpinBasis1D,
    state::AbstractVecOrMat;
    sub_sys_A=nothing,
    return_rdm=:A,
    enforce_pure::Bool=false,
    kwargs...,
)
    if _has_symmetry(basis.symmetry)
        projector = basis.symmetry.projector
        expanded = if state isa AbstractMatrix &&
                      size(state) == (length(basis), length(basis)) &&
                      !enforce_pure
            projector * state * projector'
        else
            projector * state
        end
        return partial_trace(
            _spin_parent_basis(basis),
            expanded;
            sub_sys_A,
            return_rdm,
            enforce_pure,
            kwargs...,
        )
    end
    sites_A = _subsystem_sites(basis, sub_sys_A)
    if state isa AbstractMatrix && enforce_pure &&
       size(state, 1) == length(basis)
        reductions = [
            _reduced_density_matrices(
                basis,
                @view(state[:, index]),
                sites_A,
                return_rdm,
            )
            for index in axes(state, 2)
        ]
        if return_rdm in (:both, "both")
            rho_A = cat((pair[1] for pair in reductions)...; dims=3)
            rho_B = cat((pair[2] for pair in reductions)...; dims=3)
            return rho_A, rho_B
        end
        return cat(reductions...; dims=3)
    end
    return _reduced_density_matrices(
        basis,
        state,
        sites_A,
        return_rdm,
    )
end

_density_eigenvalues(rho) =
    real.(eigvals(Hermitian((rho + rho') / 2)))

function _entropy_from_probabilities(probabilities, alpha)
    tolerance = 100 * eps(real(float(one(eltype(probabilities)))))
    probabilities = clamp.(probabilities, zero(eltype(probabilities)), Inf)
    probabilities = probabilities[probabilities .> tolerance]
    isempty(probabilities) && return zero(eltype(probabilities))
    if abs(alpha - 1) <= sqrt(eps(float(alpha)))
        return -sum(p * log(p) for p in probabilities)
    end
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    return log(sum(p^alpha for p in probabilities)) / (1 - alpha)
end

_entropy_from_density(rho, alpha) =
    _entropy_from_probabilities(_density_eigenvalues(rho), alpha)

const _SCHMIDT_SVD_CROSSOVER = 64

function _schmidt_probabilities(coefficients::AbstractMatrix)
    if min(size(coefficients)...) <= _SCHMIDT_SVD_CROSSOVER
        return abs2.(svdvals(coefficients))
    end
    gram = size(coefficients, 1) <= size(coefficients, 2) ?
        coefficients * coefficients' :
        coefficients' * coefficients
    return _density_eigenvalues(gram)
end

"""
    ent_entropy(basis, state; ...)

Von Neumann (`alpha=1`) or Rényi entanglement entropy for a subsystem.
"""
function ent_entropy(
    basis::SpinBasis1D,
    state::AbstractVecOrMat;
    sub_sys_A=nothing,
    density::Bool=true,
    return_rdm=nothing,
    enforce_pure::Bool=false,
    return_rdm_EVs::Bool=false,
    alpha::Real=1.0,
    kwargs...,
)
    sites_A = _subsystem_sites(basis, sub_sys_A)
    if state isa AbstractVector
        coefficients = _spin_pure_coefficients(basis, state, sites_A)
        schmidt_probabilities = _schmidt_probabilities(coefficients)
        normalization_A = density && !isempty(sites_A) ? length(sites_A) : 1
        sites_B = basis.L - length(sites_A)
        normalization_B = density && sites_B > 0 ? sites_B : 1
        entropy = _entropy_from_probabilities(
            schmidt_probabilities,
            alpha,
        )
        result = Dict{String,Any}(
            "Sent_A" => entropy / normalization_A,
        )
        if return_rdm in (:A, "A", :both, "both")
            result["rdm_A"] = coefficients * coefficients'
        end
        if return_rdm in (:B, "B", :both, "both")
            result["Sent_B"] = entropy / normalization_B
            result["rdm_B"] = coefficients' * coefficients
        end
        if return_rdm_EVs
            probabilities_A = collect(schmidt_probabilities)
            append!(
                probabilities_A,
                zeros(
                    eltype(probabilities_A),
                    max(0, size(coefficients, 1) - length(probabilities_A)),
                ),
            )
            sort!(probabilities_A)
            result["p_A"] = probabilities_A
        end
        return result
    end
    need_B = return_rdm in (:B, "B", :both, "both")
    rho_A, rho_B = if need_B
        partial_trace(
            basis,
            state;
            sub_sys_A=sites_A,
            return_rdm=:both,
            enforce_pure,
        )
    else
        partial_trace(
            basis,
            state;
            sub_sys_A=sites_A,
            return_rdm=:A,
            enforce_pure,
        ), nothing
    end
    normalization_A = density && !isempty(sites_A) ? length(sites_A) : 1
    sites_B = basis.L - length(sites_A)
    normalization_B = density && sites_B > 0 ? sites_B : 1
    if ndims(rho_A) == 3
        probabilities_A = [
            _density_eigenvalues(@view(rho_A[:, :, index]))
            for index in axes(rho_A, 3)
        ]
        result = Dict{String,Any}(
            "Sent_A" => [
                _entropy_from_probabilities(probabilities, alpha) /
                normalization_A
                for probabilities in probabilities_A
            ],
        )
        if return_rdm in (:A, "A", :both, "both")
            result["rdm_A"] = rho_A
        end
        if need_B
            result["Sent_B"] = [
                _entropy_from_density(
                    @view(rho_B[:, :, index]),
                    alpha,
                ) / normalization_B
                for index in axes(rho_B, 3)
            ]
            result["rdm_B"] = rho_B
        end
        return_rdm_EVs &&
            (result["p_A"] = reduce(hcat, probabilities_A))
        return result
    end
    probabilities_A = _density_eigenvalues(rho_A)
    entropy_A =
        _entropy_from_probabilities(probabilities_A, alpha) /
        normalization_A
    result = Dict{String,Any}("Sent_A" => entropy_A)
    if return_rdm in (:A, "A", :both, "both")
        result["rdm_A"] = rho_A
    end
    if need_B
        result["Sent_B"] =
            _entropy_from_density(rho_B, alpha) /
            normalization_B
        result["rdm_B"] = rho_B
    end
    if return_rdm_EVs
        result["p_A"] = probabilities_A
    end
    return result
end

"""
    coherent_state(a, n; dtype=nothing)

Return the first `n` number-state amplitudes of a harmonic-oscillator coherent
state. The recurrence avoids the `log(0)` and negative-real logarithm defects
in QuSpin's historical implementation.
"""
function coherent_state(a::Number, n::Integer; dtype::Union{Nothing,Type}=nothing)
    n >= 1 || throw(ArgumentError("n must be positive"))
    T = dtype === nothing ? promote_type(Float64, typeof(a)) : dtype
    T <: Number || throw(ArgumentError("dtype must be a numeric type"))
    result = Vector{T}(undef, n)
    if iszero(a)
        fill!(result, zero(T))
        result[1] = one(T)
        return result
    end
    magnitude = abs(a)
    phase = a / magnitude
    phase_power = one(phase)
    log_amplitude = -abs2(a) / 2
    log_magnitude = log(magnitude)
    for level in 0:(n - 1)
        result[level + 1] =
            convert(T, exp(log_amplitude) * phase_power)
        log_amplitude += log_magnitude - log(level + 1) / 2
        phase_power *= phase
    end
    return result
end

"""
    photon_Hspace_dim(N, Ntot, Nph)

Dimension of a spin-photon Hilbert space. When `Ntot` is fixed this is the
number of spin sectors with at most `Ntot` excitations; otherwise the spin
space is tensored with photon occupations `0:Nph`.
"""
function photon_Hspace_dim(
    N::Integer,
    Ntot::Union{Nothing,Integer},
    Nph::Union{Nothing,Integer},
)
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    if Ntot === nothing
        Nph === nothing &&
            throw(ArgumentError("either Ntot or Nph must be defined"))
        Nph >= 0 || throw(ArgumentError("Nph must be nonnegative"))
        dimension = (BigInt(1) << N) * (Nph + 1)
    else
        Ntot >= 0 || throw(ArgumentError("Ntot must be nonnegative"))
        dimension = sum(binomial(BigInt(N), k) for k in 0:min(N, Ntot))
    end
    return dimension <= typemax(Int) ? Int(dimension) : dimension
end

"""
    get_basis_type(N, Np, sps)

Smallest QuSpin-compatible unsigned integer type able to encode an `N`-site
state with `sps` local states. `Np` is accepted for API compatibility; it does
not reduce the number of bits needed to locate the highest occupied site.
"""
function get_basis_type(N::Integer, Np, sps::Integer)
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    sps >= 1 || throw(ArgumentError("sps must be positive"))
    max_state = BigInt(sps)^N - 1
    bits = max(1, ndigits(max_state; base=2))
    bits <= 32 && return UInt32
    bits <= 64 && return UInt64
    bits <= 256 && return UInt256
    bits <= 1024 && return UInt1024
    bits <= 4096 && return UInt4096
    bits <= 16384 && return UInt16384
    throw(ArgumentError("$bits bits exceed QuSpin's 16384-bit basis type"))
end

basis_int_to_python_int(value::Integer) = BigInt(value)
basis_int_to_python_int(values::AbstractArray{<:Integer}) =
    map(basis_int_to_python_int, values)

function python_int_to_basis_int(value::Integer; dtype::Union{Nothing,Type}=nothing)
    value >= 0 || throw(ArgumentError("value must be nonnegative"))
    T = if dtype === nothing
        bits = max(1, ndigits(BigInt(value); base=2))
        bits <= 32 ? UInt32 :
        bits <= 64 ? UInt64 :
        bits <= 256 ? UInt256 :
        bits <= 1024 ? UInt1024 :
        bits <= 4096 ? UInt4096 :
        bits <= 16384 ? UInt16384 :
        throw(ArgumentError("$bits bits exceed QuSpin's 16384-bit basis type"))
    else
        dtype
    end
    return T(value)
end

function basis_zeros(shape, dtype::Type=UInt32)
    dims = shape isa Integer ? (Int(shape),) : Tuple(Int.(shape))
    return fill(zero(dtype), dims)
end

function basis_ones(shape, dtype::Type=UInt32)
    dims = shape isa Integer ? (Int(shape),) : Tuple(Int.(shape))
    return fill(one(dtype), dims)
end

function _bitwise_result(op, arguments...; out=nothing, where=nothing)
    if where === nothing
        out === nothing && return broadcast(op, arguments...)
        broadcast!(op, out, arguments...)
        return out
    end
    result = broadcast(op, arguments...)
    result isa AbstractArray ||
        return Bool(where) ? result : (out === nothing ? zero(result) : out)
    if out === nothing
        broadcast!(
            (new, keep) -> keep ? new : zero(new),
            result,
            result,
            where,
        )
        return result
    end
    destination = out
    axes(destination) == axes(result) ||
        throw(DimensionMismatch("out and broadcast result must have the same axes"))
    broadcast!((new, keep, old) -> keep ? new : old, destination, result, where, destination)
    return destination
end

bitwise_and(x1, x2; out=nothing, where=nothing) =
    _bitwise_result(&, x1, x2; out, where)
bitwise_or(x1, x2; out=nothing, where=nothing) =
    _bitwise_result(|, x1, x2; out, where)
bitwise_xor(x1, x2; out=nothing, where=nothing) =
    _bitwise_result(xor, x1, x2; out, where)
bitwise_not(x; out=nothing, where=nothing) =
    _bitwise_result(~, x; out, where)
bitwise_leftshift(x1, x2; out=nothing, where=nothing) =
    _bitwise_result(<<, x1, x2; out, where)
bitwise_rightshift(x1, x2; out=nothing, where=nothing) =
    _bitwise_result(>>, x1, x2; out, where)

include("discrete_basis.jl")
include("composite_basis.jl")
include("user_basis.jl")

end
