module QuSpin

using LinearAlgebra

export Hamiltonian, OperatorTerm, SpinBasis1D
export eigvals, state_at, state_index, states

"""
    SpinBasis1D(L; nup=nothing, pauli=true)

Computational basis for a spin-one-half chain. Public site indices are
one-based; basis states use the low `L` bits of a `UInt64`.
"""
struct SpinBasis1D
    L::Int
    nup::Union{Nothing,Int}
    pauli::Bool
    encoded_states::Vector{UInt64}
    lookup::Dict{UInt64,Int}
end

function SpinBasis1D(
    L::Integer;
    nup::Union{Nothing,Integer}=nothing,
    pauli::Bool=true,
)
    1 <= L <= 63 || throw(ArgumentError("L must be between 1 and 63"))
    nup === nothing || 0 <= nup <= L ||
        throw(ArgumentError("nup must lie between 0 and L"))

    wanted = nup === nothing ? nothing : Int(nup)
    encoded = UInt64[]
    sizehint!(encoded, wanted === nothing ? 1 << min(Int(L), 20) : 0)
    last_state = (UInt64(1) << Int(L)) - UInt64(1)
    for state in UInt64(0):last_state
        wanted === nothing || count_ones(state) == wanted || continue
        push!(encoded, state)
    end
    lookup = Dict(state => i for (i, state) in pairs(encoded))
    return SpinBasis1D(Int(L), wanted, pauli, encoded, lookup)
end

Base.length(basis::SpinBasis1D) = length(basis.encoded_states)
states(basis::SpinBasis1D) = copy(basis.encoded_states)

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
struct Hamiltonian{T<:Number}
    basis::SpinBasis1D
    terms::Vector{OperatorTerm}
    data::Matrix{T}
end

function _coefficient_type(terms)
    coefficient_types = Type[]
    for term in terms, coupling in term.couplings
        push!(coefficient_types, typeof(first(coupling)))
    end
    isempty(coefficient_types) && return Float64
    return promote_type(Float64, coefficient_types...)
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
    else
        throw(ArgumentError("unsupported spin operator '$op'"))
    end
end

function _assemble(basis::SpinBasis1D, terms::Vector{OperatorTerm}, ::Type{T}) where {T}
    matrix = zeros(T, length(basis), length(basis))
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
            matrix[row, column] += amplitude
        end
    end
    return matrix
end

function Hamiltonian(basis::SpinBasis1D, terms::AbstractVector{<:OperatorTerm})
    normalized = OperatorTerm[terms...]
    T = _coefficient_type(normalized)
    return Hamiltonian{T}(basis, normalized, _assemble(basis, normalized, T))
end

Base.size(H::Hamiltonian) = size(H.data)
Base.getindex(H::Hamiltonian, indices...) = getindex(H.data, indices...)
Base.Matrix(H::Hamiltonian) = copy(H.data)
Base.:*(H::Hamiltonian, vector::AbstractVecOrMat) = H.data * vector
LinearAlgebra.ishermitian(H::Hamiltonian) = ishermitian(H.data)

function LinearAlgebra.eigvals(H::Hamiltonian)
    return ishermitian(H) ? eigvals(Hermitian(H.data)) : eigvals(H.data)
end

end
