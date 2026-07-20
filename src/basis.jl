module Basis

using LinearAlgebra
using SparseArrays

export AbstractBasis, FixedUInt, SpinBasis1D, SpinBasisGeneral
export BosonBasis1D, BosonBasisGeneral
export SpinlessFermionBasis1D, SpinlessFermionBasisGeneral
export SpinfulFermionBasis1D, SpinfulFermionBasisGeneral
export TensorBasis, PhotonBasis, UserBasis
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

abstract type AbstractBasis end

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
Base.zero(::Type{FixedUInt{W}}) where {W} = FixedUInt{W}(0)
Base.one(::Type{FixedUInt{W}}) where {W} = FixedUInt{W}(1)
Base.typemin(::Type{FixedUInt{W}}) where {W} = zero(FixedUInt{W})
Base.typemax(::Type{FixedUInt{W}}) where {W} = FixedUInt{W}((BigInt(1) << W) - 1)
Base.iszero(value::FixedUInt) = iszero(value.value)
Base.isone(value::FixedUInt) = isone(value.value)
Base.:(==)(left::FixedUInt, right::FixedUInt) = left.value == right.value
Base.:(==)(left::FixedUInt, right::Integer) = left.value == right
Base.:(==)(left::Integer, right::FixedUInt) = left == right.value
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
Base.:&(left::FixedUInt{W}, right::FixedUInt{W}) where {W} =
    FixedUInt{W}(left.value & right.value)
Base.:|(left::FixedUInt{W}, right::Integer) where {W} = _fixed_binary(|, left, right)
Base.:|(left::Integer, right::FixedUInt{W}) where {W} = right | left
Base.:|(left::FixedUInt{W}, right::FixedUInt{W}) where {W} =
    FixedUInt{W}(left.value | right.value)
Base.xor(left::FixedUInt{W}, right::Integer) where {W} = _fixed_binary(xor, left, right)
Base.xor(left::Integer, right::FixedUInt{W}) where {W} = xor(right, left)
Base.xor(left::FixedUInt{W}, right::FixedUInt{W}) where {W} =
    FixedUInt{W}(xor(left.value, right.value))
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

const SpinBasisGeneral = SpinBasis1D

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
    1 <= L <= 63 || throw(ArgumentError("L must be between 1 and 63"))
    S in ("1/2", "0.5", 0.5, 1 // 2) ||
        throw(ArgumentError("SpinBasis1D currently supports spin one-half"))
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
    encoded = UInt64[]
    sizehint!(encoded, wanted === nothing ? 1 << min(Int(L), 20) : 0)
    last_state = (UInt64(1) << Int(L)) - UInt64(1)
    for state in UInt64(0):last_state
        wanted === nothing || count_ones(state) == wanted || continue
        push!(encoded, state)
    end
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
        symmetry = _identity_symmetry_data(encoded, blocks)
        return SpinBasis1D(Int(L), wanted, pauli, encoded, lookup, symmetry)
    end

    a > 0 && L % a == 0 ||
        throw(ArgumentError("a must be a positive divisor of L"))
    order = Int(L ÷ a)
    projector = sparse(
        collect(1:length(encoded)),
        collect(1:length(encoded)),
        ones(ComplexF64, length(encoded)),
        length(encoded),
        length(encoded),
    )

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
        symmetry_matrix = _signed_permutation(encoded, lookup, transform)
        projector = _intersect_eigenspace(
            projector,
            symmetry_matrix,
            ComplexF64(value),
        )
        blocks[name] = Int(value)
    end

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

function projection_matrix(
    basis::SpinBasis1D,
    ::Type{T}=Float64,
) where {T<:Number}
    return _full_projection_matrix(
        basis.encoded_states,
        basis.symmetry,
        1 << basis.L,
        T,
    )
end

function project_from(
    basis::SpinBasis1D,
    vector::AbstractVecOrMat;
    sparse::Bool=true,
    pcon::Bool=false,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    return projection_matrix(basis, eltype(vector)) * vector
end

get_vec(basis::SpinBasis1D, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)

expanded_form(basis::SpinBasis1D, static=Any[], dynamic=Any[]) =
    (static, dynamic)

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

function operator_matrix(
    basis::SpinBasis1D,
    opstring::AbstractString,
    couplings,
)
    if _has_symmetry(basis.symmetry)
        parent_symmetry = _identity_symmetry_data(
            basis.symmetry.parent_states,
            Dict(:nup => basis.nup, :pauli => basis.pauli),
        )
        parent = SpinBasis1D(
            basis.L,
            basis.nup,
            basis.pauli,
            copy(basis.symmetry.parent_states),
            copy(basis.symmetry.parent_lookup),
            parent_symmetry,
        )
        parent_matrix = operator_matrix(parent, opstring, couplings)
        return Matrix(basis.symmetry.projector' * parent_matrix *
                      basis.symmetry.projector)
    end
    matrix = zeros(ComplexF64, length(basis), length(basis))
    for coupling in couplings
        length(coupling) == length(opstring) + 1 ||
            throw(ArgumentError("operator arity and sites differ"))
        for (column, initial) in pairs(basis.encoded_states)
            state = initial
            amplitude = complex(first(coupling))
            alive = true
            for (op, site) in Iterators.reverse(
                collect(zip(opstring, coupling[2:end])),
            )
                state, factor, alive = _apply_spin_local(basis, state, op, site)
                alive || break
                amplitude *= factor
            end
            alive || continue
            row = get(basis.lookup, state, 0)
            row == 0 || (matrix[row, column] += amplitude)
        end
    end
    return matrix
end

function inplace_op!(out, basis::SpinBasis1D, opstring, couplings)
    size(out) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("out must have shape (Ns,Ns)"))
    out .+= operator_matrix(basis, opstring, couplings)
    return out
end

op_bra_ket(basis::SpinBasis1D, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)

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
    return coefficients * coefficients', coefficients' * coefficients
end

function _reduced_density_matrices(
    basis::SpinBasis1D,
    state::AbstractMatrix,
    sites_A,
)
    size(state) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("density matrix must match the basis dimension"))
    sites_B = setdiff(collect(1:basis.L), sites_A)
    rho_A = zeros(eltype(state), 1 << length(sites_A), 1 << length(sites_A))
    rho_B = zeros(eltype(state), 1 << length(sites_B), 1 << length(sites_B))
    for (row_position, row_state) in pairs(basis.encoded_states)
        row_A = _subsystem_index(row_state, sites_A)
        row_B = _subsystem_index(row_state, sites_B)
        for (column_position, column_state) in pairs(basis.encoded_states)
            column_A = _subsystem_index(column_state, sites_A)
            column_B = _subsystem_index(column_state, sites_B)
            value = state[row_position, column_position]
            row_B == column_B && (rho_A[row_A, column_A] += value)
            row_A == column_A && (rho_B[row_B, column_B] += value)
        end
    end
    return rho_A, rho_B
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
        projector = projection_matrix(basis, ComplexF64)
        expanded = if state isa AbstractMatrix &&
                      size(state) == (length(basis), length(basis)) &&
                      !enforce_pure
            projector * state * projector'
        else
            projector * state
        end
        full_basis = SpinBasis1D(basis.L; pauli=basis.pauli)
        return partial_trace(
            full_basis,
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
            _reduced_density_matrices(basis, @view(state[:, index]), sites_A)
            for index in axes(state, 2)
        ]
        rho_A = cat((pair[1] for pair in reductions)...; dims=3)
        rho_B = cat((pair[2] for pair in reductions)...; dims=3)
    else
        rho_A, rho_B = _reduced_density_matrices(basis, state, sites_A)
    end
    return return_rdm in (:A, "A") ? rho_A :
           return_rdm in (:B, "B") ? rho_B :
           return_rdm in (:both, "both") ? (rho_A, rho_B) :
           throw(ArgumentError("return_rdm must be A, B, or both"))
end

function _entropy_from_density(rho, alpha)
    probabilities = real.(eigvals(Hermitian((rho + rho') / 2)))
    tolerance = 100 * eps(real(float(one(eltype(probabilities)))))
    probabilities = clamp.(probabilities, zero(eltype(probabilities)), Inf)
    probabilities = probabilities[probabilities .> tolerance]
    isempty(probabilities) && return zero(eltype(probabilities))
    alpha == 1 && return -sum(p * log(p) for p in probabilities)
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    return log(sum(p^alpha for p in probabilities)) / (1 - alpha)
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
    rho_A, rho_B = partial_trace(
        basis,
        state;
        sub_sys_A=sites_A,
        return_rdm=:both,
        enforce_pure,
    )
    ndims(rho_A) == 2 ||
        throw(ArgumentError("batched pure-state entropy is not yet returned as one call"))
    normalization_A = density && !isempty(sites_A) ? length(sites_A) : 1
    sites_B = basis.L - length(sites_A)
    normalization_B = density && sites_B > 0 ? sites_B : 1
    entropy_A = _entropy_from_density(rho_A, alpha) / normalization_A
    result = Dict{String,Any}("Sent_A" => entropy_A)
    if return_rdm in (:A, "A", :both, "both")
        result["rdm_A"] = rho_A
    end
    if return_rdm in (:B, "B", :both, "both")
        result["Sent_B"] = _entropy_from_density(rho_B, alpha) / normalization_B
        result["rdm_B"] = rho_B
    end
    if return_rdm_EVs
        result["p_A"] = real.(eigvals(Hermitian((rho_A + rho_A') / 2)))
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
    result[1] = convert(T, exp(-abs2(a) / 2))
    for level in 1:(n - 1)
        result[level + 1] = result[level] * a / sqrt(level)
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
    result = broadcast(op, arguments...)
    if where === nothing
        out === nothing && return result
        copyto!(out, result)
        return out
    end
    result isa AbstractArray ||
        return Bool(where) ? result : (out === nothing ? zero(result) : out)
    destination = out === nothing ? fill!(similar(result), zero(eltype(result))) : out
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
