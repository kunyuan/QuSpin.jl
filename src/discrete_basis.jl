"""
Shared finite local-state basis used by bosons and fermions. States are encoded
as base-`sps` integers with site one in the least-significant digit.
"""
struct DiscreteBasis{K} <: AbstractBasis
    L::Int
    sps::Int
    conservation::Any
    encoded_states::Vector{UInt64}
    occupations::Matrix{Int}
    lookup::Dict{UInt64,Int}
    description::String
    operators::Tuple
end

const BosonBasis1D = DiscreteBasis{:boson}
const BosonBasisGeneral = DiscreteBasis{:boson}
const SpinlessFermionBasis1D = DiscreteBasis{:spinless_fermion}
const SpinlessFermionBasisGeneral = DiscreteBasis{:spinless_fermion}
const SpinfulFermionBasis1D = DiscreteBasis{:spinful_fermion}
const SpinfulFermionBasisGeneral = DiscreteBasis{:spinful_fermion}

function _digits(value::Integer, L::Integer, sps::Integer)
    remaining = Int(value)
    result = Vector{Int}(undef, L)
    for site in 1:L
        result[site] = mod(remaining, sps)
        remaining = div(remaining, sps)
    end
    return result
end

function _make_discrete_basis(
    ::Val{K},
    L::Integer,
    sps::Integer,
    conservation,
    keep,
    description,
    operators,
) where {K}
    L > 0 || throw(ArgumentError("L must be positive"))
    sps > 1 || throw(ArgumentError("sps must exceed one"))
    dimension = BigInt(sps)^L
    dimension <= typemax(UInt64) ||
        throw(ArgumentError("basis encoding exceeds UInt64"))
    encoded = UInt64[]
    rows = Vector{Vector{Int}}()
    for value in UInt64(0):(UInt64(dimension) - 1)
        occupations = _digits(value, L, sps)
        keep(occupations) || continue
        push!(encoded, value)
        push!(rows, occupations)
    end
    occupation_matrix = isempty(rows) ?
        Matrix{Int}(undef, 0, L) :
        reduce(vcat, permutedims.(rows))
    return DiscreteBasis{K}(
        Int(L),
        Int(sps),
        conservation,
        encoded,
        occupation_matrix,
        Dict(state => index for (index, state) in pairs(encoded)),
        description,
        Tuple(operators),
    )
end

function DiscreteBasis{:boson}(
    L::Integer;
    Nb=nothing,
    sps::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    maximum_particles = Nb === nothing ?
        nothing :
        Nb isa Integer ? Int(Nb) : maximum(Int.(collect(Nb)))
    local_states = sps === nothing ?
        maximum_particles === nothing ?
            throw(ArgumentError("boson basis requires Nb or sps")) :
            maximum_particles + 1 :
        Int(sps)
    wanted = Nb === nothing ?
        nothing :
        Set(Nb isa Integer ? [Int(Nb)] : Int.(collect(Nb)))
    keep = occupations -> wanted === nothing || sum(occupations) in wanted
    return _make_discrete_basis(
        Val(:boson),
        L,
        local_states,
        Nb,
        keep,
        "boson lattice basis",
        ("I", "+", "-", "n", "z"),
    )
end

function DiscreteBasis{:spinless_fermion}(
    L::Integer;
    Nf=nothing,
    kwargs...,
)
    wanted = Nf === nothing ?
        nothing :
        Set(Nf isa Integer ? [Int(Nf)] : Int.(collect(Nf)))
    keep = occupations -> wanted === nothing || sum(occupations) in wanted
    return _make_discrete_basis(
        Val(:spinless_fermion),
        L,
        2,
        Nf,
        keep,
        "spinless fermion lattice basis",
        ("I", "+", "-", "n", "z"),
    )
end

function DiscreteBasis{:spinful_fermion}(
    L::Integer;
    Nf=nothing,
    kwargs...,
)
    wanted = Nf === nothing ? nothing : Tuple(Int.(Nf))
    keep = occupations -> wanted === nothing || (
        count(digit -> digit & 1 == 1, occupations) == wanted[1] &&
        count(digit -> digit & 2 == 2, occupations) == wanted[2]
    )
    return _make_discrete_basis(
        Val(:spinful_fermion),
        L,
        4,
        Nf,
        keep,
        "spinful fermion lattice basis",
        ("I", "+", "-", "n", "z", "|"),
    )
end

Base.length(basis::DiscreteBasis) = length(basis.encoded_states)
Base.:(==)(left::DiscreteBasis{K}, right::DiscreteBasis{K}) where {K} =
    left.L == right.L &&
    left.sps == right.sps &&
    left.conservation == right.conservation &&
    left.encoded_states == right.encoded_states

function Base.getproperty(basis::DiscreteBasis{K}, name::Symbol) where {K}
    name in (:N, :L) && return getfield(basis, :L)
    name === :Ns && return length(getfield(basis, :encoded_states))
    name === :blocks && return Dict(:conservation => getfield(basis, :conservation))
    name === :dtype && return UInt64
    name === :noncommuting_bits && return K in (:spinless_fermion, :spinful_fermion) ?
        [(collect(1:getfield(basis, :L)), -1)] :
        Tuple{Vector{Int},Int}[]
    name === :states && return copy(getfield(basis, :encoded_states))
    return getfield(basis, name)
end

states(basis::DiscreteBasis) = copy(basis.encoded_states)

function projection_matrix(
    basis::DiscreteBasis,
    ::Type{T}=Float64,
) where {T<:Number}
    projector = zeros(T, basis.sps^basis.L, length(basis))
    for (column, state) in pairs(basis.encoded_states)
        projector[Int(state) + 1, column] = one(T)
    end
    return projector
end

function project_from(
    basis::DiscreteBasis,
    vector::AbstractVecOrMat;
    sparse::Bool=true,
    pcon::Bool=false,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    return projection_matrix(basis, eltype(vector)) * vector
end

get_vec(basis::DiscreteBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)
expanded_form(basis::DiscreteBasis, static=Any[], dynamic=Any[]) =
    (static, dynamic)

function state_index(basis::DiscreteBasis, state::Integer)
    state >= 0 || throw(ArgumentError("state must be nonnegative"))
    return get(basis.lookup, UInt64(state)) do
        throw(ArgumentError("state $state is not represented by this basis"))
    end
end

function state_at(basis::DiscreteBasis, index::Integer)
    checkbounds(basis.encoded_states, index)
    return basis.encoded_states[index]
end

function int_to_state(
    basis::DiscreteBasis,
    state::Integer;
    bracket_notation::Bool=true,
)
    0 <= state < basis.sps^basis.L ||
        throw(ArgumentError("state lies outside the full local-state space"))
    digits = reverse(_digits(state, basis.L, basis.sps))
    text = join(digits, " ")
    return bracket_notation ? "|$text>" : replace(text, " " => "")
end

function state_to_int(basis::DiscreteBasis, state::AbstractString)
    compact = strip(state, ['|', '>'])
    tokens = occursin(' ', compact) ?
        split(compact) :
        collect(compact)
    length(tokens) == basis.L ||
        throw(ArgumentError("state must contain exactly $(basis.L) occupations"))
    occupations = parse.(Int, string.(tokens))
    all(0 .<= occupations .< basis.sps) ||
        throw(ArgumentError("local occupations must lie in 0:$(basis.sps - 1)"))
    return UInt64(sum(occupations[end - site + 1] * basis.sps^(site - 1) for site in 1:basis.L))
end

function _discrete_subsystem_sites(basis::DiscreteBasis, sub_sys_A)
    sites = sub_sys_A === nothing ?
        collect(1:fld(basis.L, 2)) :
        Int.(collect(sub_sys_A))
    allunique(sites) || throw(ArgumentError("subsystem sites must be unique"))
    all(site -> 1 <= site <= basis.L, sites) ||
        throw(ArgumentError("subsystem sites must lie in 1:$(basis.L)"))
    return sites
end

function _local_index(occupations, sites, sps)
    index = 0
    multiplier = 1
    for site in sites
        index += occupations[site] * multiplier
        multiplier *= sps
    end
    return index + 1
end

function _discrete_reductions(
    basis::DiscreteBasis,
    state::AbstractVector,
    sites_A,
)
    length(state) == length(basis) ||
        throw(DimensionMismatch("state length must equal Ns"))
    sites_B = setdiff(collect(1:basis.L), sites_A)
    coefficients = zeros(
        eltype(state),
        basis.sps^length(sites_A),
        basis.sps^length(sites_B),
    )
    for row in axes(basis.occupations, 1)
        index_A = _local_index(@view(basis.occupations[row, :]), sites_A, basis.sps)
        index_B = _local_index(@view(basis.occupations[row, :]), sites_B, basis.sps)
        coefficients[index_A, index_B] = state[row]
    end
    return coefficients * coefficients', coefficients' * coefficients
end

function _discrete_reductions(
    basis::DiscreteBasis,
    state::AbstractMatrix,
    sites_A,
)
    size(state) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("density matrix must match Ns"))
    sites_B = setdiff(collect(1:basis.L), sites_A)
    rho_A = zeros(eltype(state), basis.sps^length(sites_A), basis.sps^length(sites_A))
    rho_B = zeros(eltype(state), basis.sps^length(sites_B), basis.sps^length(sites_B))
    for row in axes(state, 1), column in axes(state, 2)
        row_occ = @view basis.occupations[row, :]
        column_occ = @view basis.occupations[column, :]
        row_A = _local_index(row_occ, sites_A, basis.sps)
        column_A = _local_index(column_occ, sites_A, basis.sps)
        row_B = _local_index(row_occ, sites_B, basis.sps)
        column_B = _local_index(column_occ, sites_B, basis.sps)
        row_B == column_B && (rho_A[row_A, column_A] += state[row, column])
        row_A == column_A && (rho_B[row_B, column_B] += state[row, column])
    end
    return rho_A, rho_B
end

function partial_trace(
    basis::DiscreteBasis,
    state::AbstractVecOrMat;
    sub_sys_A=nothing,
    return_rdm=:A,
    enforce_pure::Bool=false,
    kwargs...,
)
    sites_A = _discrete_subsystem_sites(basis, sub_sys_A)
    rho_A, rho_B = _discrete_reductions(basis, state, sites_A)
    return return_rdm in (:A, "A") ? rho_A :
           return_rdm in (:B, "B") ? rho_B :
           return_rdm in (:both, "both") ? (rho_A, rho_B) :
           throw(ArgumentError("return_rdm must be A, B, or both"))
end

function ent_entropy(
    basis::DiscreteBasis,
    state::AbstractVecOrMat;
    sub_sys_A=nothing,
    density::Bool=true,
    return_rdm=nothing,
    alpha::Real=1.0,
    kwargs...,
)
    sites_A = _discrete_subsystem_sites(basis, sub_sys_A)
    rho_A, rho_B = partial_trace(
        basis,
        state;
        sub_sys_A=sites_A,
        return_rdm=:both,
    )
    normalization_A = density && !isempty(sites_A) ? length(sites_A) : 1
    sites_B = basis.L - length(sites_A)
    normalization_B = density && sites_B > 0 ? sites_B : 1
    result = Dict{String,Any}(
        "Sent_A" => _entropy_from_density(rho_A, alpha) / normalization_A,
    )
    if return_rdm in (:A, "A", :both, "both")
        result["rdm_A"] = rho_A
    end
    if return_rdm in (:B, "B", :both, "both")
        result["Sent_B"] = _entropy_from_density(rho_B, alpha) / normalization_B
        result["rdm_B"] = rho_B
    end
    return result
end

function _apply_discrete_local(basis::DiscreteBasis{K}, occupations, op, site) where {K}
    value = occupations[site]
    if op == 'I'
        return one(Float64), true
    elseif op == 'n'
        return float(K === :spinful_fermion ? count_ones(value) : value), true
    elseif op == 'z'
        return value - (basis.sps - 1) / 2, true
    elseif op == '+'
        value < basis.sps - 1 || return 0.0, false
        sign = K === :spinless_fermion ? (-1)^sum(occupations[1:(site - 1)]) : 1
        occupations[site] += 1
        factor = K === :boson ? sqrt(value + 1) : 1
        return sign * factor, true
    elseif op == '-'
        value > 0 || return 0.0, false
        sign = K === :spinless_fermion ? (-1)^sum(occupations[1:(site - 1)]) : 1
        occupations[site] -= 1
        factor = K === :boson ? sqrt(value) : 1
        return sign * factor, true
    end
    throw(ArgumentError("unsupported local operator '$op'"))
end

function operator_matrix(
    basis::DiscreteBasis,
    opstring::AbstractString,
    couplings,
)
    matrix = zeros(ComplexF64, length(basis), length(basis))
    for coupling in couplings
        coefficient = first(coupling)
        sites = coupling[2:end]
        length(sites) == length(opstring) ||
            throw(ArgumentError("operator arity and sites differ"))
        for column in axes(basis.occupations, 1)
            occupations = collect(@view basis.occupations[column, :])
            amplitude = complex(coefficient)
            alive = true
            for (op, site) in zip(opstring, sites)
                factor, alive = _apply_discrete_local(basis, occupations, op, site)
                alive || break
                amplitude *= factor
            end
            alive || continue
            encoded = UInt64(sum(
                occupations[site] * basis.sps^(site - 1)
                for site in 1:basis.L
            ))
            row = get(basis.lookup, encoded, 0)
            row == 0 || (matrix[row, column] += amplitude)
        end
    end
    return matrix
end

function inplace_op!(out, basis::DiscreteBasis, opstring, couplings)
    size(out) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("out must have shape (Ns,Ns)"))
    out .+= operator_matrix(basis, opstring, couplings)
    return out
end

check_hermitian(basis::AbstractBasis, static, dynamic=Any[]) = true
check_pcon(basis::AbstractBasis, static, dynamic=Any[]) = true
check_symm(basis::AbstractBasis, static, dynamic=Any[]) = true
representative(basis::AbstractBasis, state::Integer) = state
normalization(basis::AbstractBasis, state::Integer) = one(Float64)
get_amp(basis::AbstractBasis, state::Integer) = one(Float64)
make_basis!(basis::AbstractBasis) = basis
make_basis_blocks(basis::AbstractBasis) = [1:length(basis)]
project_to(basis::AbstractBasis, vector::AbstractVecOrMat) =
    projection_matrix(basis)' * vector
op_bra_ket(basis::DiscreteBasis, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)
op_shift_sector(target::AbstractBasis, source::AbstractBasis, operator, vector) =
    projection_matrix(target)' * projection_matrix(source) * vector
