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
    symmetry::SymmetryData
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

function _discrete_basis_from_encoded(
    ::Val{K},
    L::Integer,
    sps::Integer,
    conservation,
    encoded::Vector{UInt64},
    description,
    operators,
) where {K}
    occupations = Matrix{Int}(undef, length(encoded), L)
    for (row, value) in pairs(encoded)
        remaining = value
        for site in 1:L
            occupations[row, site] = Int(rem(remaining, UInt64(sps)))
            remaining = div(remaining, UInt64(sps))
        end
    end
    blocks = Dict{Symbol,Any}(:conservation => conservation)
    lookup = Dict(state => index for (index, state) in pairs(encoded))
    symmetry = _identity_symmetry_data(encoded, blocks, lookup)
    return DiscreteBasis{K}(
        Int(L),
        Int(sps),
        conservation,
        encoded,
        occupations,
        lookup,
        description,
        Tuple(operators),
        symmetry,
    )
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
    for value in UInt64(0):(UInt64(dimension) - 1)
        occupations = _digits(value, L, sps)
        keep(occupations) || continue
        push!(encoded, value)
    end
    return _discrete_basis_from_encoded(
        Val(K),
        L,
        sps,
        conservation,
        encoded,
        description,
        operators,
    )
end

function _bounded_composition_states(
    L::Int,
    sps::Int,
    totals,
)
    encoded = UInt64[]
    weights = UInt64[UInt64(sps)^(site - 1) for site in 1:L]
    function append_states!(site::Int, remaining::Int, value::UInt64)
        if site > L
            iszero(remaining) && push!(encoded, value)
            return
        end
        sites_left = L - site
        minimum_here = max(0, remaining - sites_left * (sps - 1))
        maximum_here = min(sps - 1, remaining)
        for occupation in minimum_here:maximum_here
            append_states!(
                site + 1,
                remaining - occupation,
                value + UInt64(occupation) * weights[site],
            )
        end
    end
    for total in sort!(collect(totals))
        0 <= total <= L * (sps - 1) || continue
        append_states!(1, total, zero(UInt64))
    end
    sort!(encoded)
    return encoded
end

function _discrete_symmetry_basis(
    basis::DiscreteBasis{K};
    a::Integer=1,
    kblock=nothing,
    pblock=nothing,
    cblock=nothing,
    pcblock=nothing,
    cAblock=nothing,
    cBblock=nothing,
    sblock=nothing,
    psblock=nothing,
) where {K}
    requested = any(value !== nothing for value in (
        kblock,
        pblock,
        cblock,
        pcblock,
        cAblock,
        cBblock,
        sblock,
        psblock,
    ))
    requested || return basis
    a > 0 && basis.L % a == 0 ||
        throw(ArgumentError("a must be a positive divisor of L"))

    parent_states = basis.encoded_states
    lookup = basis.lookup
    dimension = length(parent_states)
    projector = sparse(
        collect(1:dimension),
        collect(1:dimension),
        ones(ComplexF64, dimension),
        dimension,
        dimension,
    )
    blocks = copy(basis.symmetry.blocks)
    order = basis.L ÷ Int(a)
    translation = [mod1(site + Int(a), basis.L) for site in 1:basis.L]
    parity = [basis.L - site + 1 for site in 1:basis.L]
    fermionic = K in (:spinless_fermion, :spinful_fermion)
    spinful = K === :spinful_fermion

    function site_transform(permutation)
        return state -> _site_permutation_transform(
            state,
            _digits(state, basis.L, basis.sps),
            basis.sps,
            permutation;
            fermionic,
            spinful,
        )
    end

    if kblock !== nothing
        momentum = mod(Int(kblock), order)
        projector = _cyclic_projector(
            parent_states,
            lookup,
            site_transform(translation),
            ComplexF64(cis(2π * momentum / order)),
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

    parity_transform = site_transform(parity)
    complement_transform = state -> begin
        occupations = _digits(state, basis.L, basis.sps)
        complemented = basis.sps - 1 .- occupations
        encoded = UInt64(sum(
            complemented[site] * basis.sps^(site - 1)
            for site in 1:basis.L
        ))
        encoded, 1.0 + 0im
    end
    cA_transform = state -> begin
        occupations = _digits(state, basis.L, basis.sps)
        for site in 1:2:basis.L
            occupations[site] = basis.sps - 1 - occupations[site]
        end
        encoded = UInt64(sum(
            occupations[site] * basis.sps^(site - 1)
            for site in 1:basis.L
        ))
        encoded, 1.0 + 0im
    end
    cB_transform = state -> begin
        occupations = _digits(state, basis.L, basis.sps)
        for site in 2:2:basis.L
            occupations[site] = basis.sps - 1 - occupations[site]
        end
        encoded = UInt64(sum(
            occupations[site] * basis.sps^(site - 1)
            for site in 1:basis.L
        ))
        encoded, 1.0 + 0im
    end
    spin_swap_transform = state -> begin
        K === :spinful_fermion ||
            throw(ArgumentError("sblock is defined only for spinful fermions"))
        occupations = _digits(state, basis.L, basis.sps)
        transformed = [((digit & 1) << 1) | ((digit & 2) >> 1) for digit in occupations]
        occupied = Int[]
        for site in 1:basis.L
            occupations[site] & 1 == 1 && push!(occupied, site)
            occupations[site] & 2 == 2 && push!(occupied, basis.L + site)
        end
        sort!(occupied)
        permutation = vcat(
            collect((basis.L + 1):(2 * basis.L)),
            collect(1:basis.L),
        )
        phase = _permutation_phase(occupied, permutation)
        encoded = UInt64(sum(
            transformed[site] * basis.sps^(site - 1)
            for site in 1:basis.L
        ))
        encoded, phase
    end
    compose(first_transform, second_transform) = state -> begin
        intermediate, first_phase = first_transform(state)
        transformed, second_phase = second_transform(intermediate)
        transformed, first_phase * second_phase
    end

    constraints = Tuple{Symbol,Any,Any}[
        (:pblock, pblock, parity_transform),
    ]
    if K === :boson
        append!(constraints, [
            (:cblock, cblock, complement_transform),
            (:pcblock, pcblock, compose(parity_transform, complement_transform)),
            (:cAblock, cAblock, cA_transform),
            (:cBblock, cBblock, cB_transform),
        ])
    elseif K === :spinful_fermion
        append!(constraints, [
            (:sblock, sblock, spin_swap_transform),
            (:psblock, psblock, compose(parity_transform, spin_swap_transform)),
        ])
    elseif any(value !== nothing for value in (cblock, pcblock, cAblock, cBblock, sblock, psblock))
        throw(ArgumentError("requested symmetry is not defined for spinless fermions"))
    end

    for (name, value, transform) in constraints
        value === nothing && continue
        value in (-1, 1) ||
            throw(ArgumentError("$name must be +1 or -1"))
        symmetry_matrix = _signed_permutation(parent_states, lookup, transform)
        projector = _intersect_eigenspace(
            projector,
            symmetry_matrix,
            ComplexF64(value),
        )
        blocks[name] = Int(value)
    end

    symmetry = _finalize_symmetry_data(parent_states, projector, blocks)
    representatives = _representative_states(parent_states, symmetry.projector)
    occupations = isempty(representatives) ?
        Matrix{Int}(undef, 0, basis.L) :
        reduce(vcat, permutedims.(
            _digits(state, basis.L, basis.sps) for state in representatives
        ))
    return DiscreteBasis{K}(
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
end

function DiscreteBasis{:boson}(
    L::Integer;
    Nb=nothing,
    nb=nothing,
    sps::Union{Nothing,Integer}=nothing,
    a::Integer=1,
    kblock=nothing,
    pblock=nothing,
    cblock=nothing,
    pcblock=nothing,
    cAblock=nothing,
    cBblock=nothing,
)
    Nb !== nothing && nb !== nothing &&
        throw(ArgumentError("specify only one of Nb and nb"))
    selected_particles = Nb === nothing && nb !== nothing ?
        round(Int, float(nb) * L) :
        Nb
    maximum_particles = selected_particles === nothing ?
        nothing :
        selected_particles isa Integer ?
            Int(selected_particles) :
            maximum(Int.(collect(selected_particles)))
    local_states = sps === nothing ?
        maximum_particles === nothing ?
            throw(ArgumentError("boson basis requires Nb or sps")) :
            maximum_particles + 1 :
        Int(sps)
    wanted = selected_particles === nothing ?
        nothing :
        Set(selected_particles isa Integer ?
            [Int(selected_particles)] :
            Int.(collect(selected_particles)))
    keep = occupations -> wanted === nothing || sum(occupations) in wanted
    basis = if wanted === nothing
        _make_discrete_basis(
            Val(:boson),
            L,
            local_states,
            selected_particles,
            keep,
            "boson lattice basis",
            ("I", "+", "-", "n", "z"),
        )
    else
        _discrete_basis_from_encoded(
            Val(:boson),
            L,
            local_states,
            selected_particles,
            _bounded_composition_states(Int(L), local_states, wanted),
            "boson lattice basis",
            ("I", "+", "-", "n", "z"),
        )
    end
    return _discrete_symmetry_basis(
        basis;
        a,
        kblock,
        pblock,
        cblock,
        pcblock,
        cAblock,
        cBblock,
    )
end

function DiscreteBasis{:spinless_fermion}(
    L::Integer;
    Nf=nothing,
    nf=nothing,
    a::Integer=1,
    kblock=nothing,
    pblock=nothing,
)
    Nf !== nothing && nf !== nothing &&
        throw(ArgumentError("specify only one of Nf and nf"))
    selected_particles = Nf === nothing && nf !== nothing ?
        round(Int, float(nf) * L) :
        Nf
    wanted = selected_particles === nothing ?
        nothing :
        Set(selected_particles isa Integer ?
            [Int(selected_particles)] :
            Int.(collect(selected_particles)))
    keep = occupations -> wanted === nothing || sum(occupations) in wanted
    basis = if wanted === nothing
        _make_discrete_basis(
            Val(:spinless_fermion),
            L,
            2,
            selected_particles,
            keep,
            "spinless fermion lattice basis",
            ("I", "+", "-", "n", "z"),
        )
    else
        encoded = UInt64[]
        for particles in sort!(collect(wanted))
            0 <= particles <= L || continue
            append!(encoded, _fixed_weight_states(Int(L), particles))
        end
        sort!(encoded)
        _discrete_basis_from_encoded(
            Val(:spinless_fermion),
            L,
            2,
            selected_particles,
            encoded,
            "spinless fermion lattice basis",
            ("I", "+", "-", "n", "z"),
        )
    end
    return _discrete_symmetry_basis(basis; a, kblock, pblock)
end

function DiscreteBasis{:spinful_fermion}(
    L::Integer;
    Nf=nothing,
    nf=nothing,
    a::Integer=1,
    kblock=nothing,
    pblock=nothing,
    sblock=nothing,
    psblock=nothing,
)
    Nf !== nothing && nf !== nothing &&
        throw(ArgumentError("specify only one of Nf and nf"))
    selected_particles = if Nf !== nothing
        Nf
    elseif nf !== nothing
        Tuple(round.(Int, float.(nf) .* L))
    else
        nothing
    end
    wanted = selected_particles === nothing ?
        nothing :
        Tuple(Int.(selected_particles))
    keep = occupations -> wanted === nothing || (
        count(digit -> digit & 1 == 1, occupations) == wanted[1] &&
        count(digit -> digit & 2 == 2, occupations) == wanted[2]
    )
    basis = if wanted === nothing
        _make_discrete_basis(
            Val(:spinful_fermion),
            L,
            4,
            selected_particles,
            keep,
            "spinful fermion lattice basis",
            ("I", "+", "-", "n", "z", "|"),
        )
    elseif all(particles -> 0 <= particles <= L, wanted)
        up_states = _fixed_weight_states(Int(L), wanted[1])
        down_states = _fixed_weight_states(Int(L), wanted[2])
        encoded = Vector{UInt64}(undef, length(up_states) * length(down_states))
        index = 0
        for up in up_states, down in down_states
            value = zero(UInt64)
            for site in 0:(Int(L) - 1)
                up_bit = (up >> site) & UInt64(1)
                down_bit = (down >> site) & UInt64(1)
                value |= (up_bit | (down_bit << 1)) << (2 * site)
            end
            index += 1
            encoded[index] = value
        end
        sort!(encoded)
        _discrete_basis_from_encoded(
            Val(:spinful_fermion),
            L,
            4,
            selected_particles,
            encoded,
            "spinful fermion lattice basis",
            ("I", "+", "-", "n", "z", "|"),
        )
    else
        _discrete_basis_from_encoded(
            Val(:spinful_fermion),
            L,
            4,
            selected_particles,
            UInt64[],
            "spinful fermion lattice basis",
            ("I", "+", "-", "n", "z", "|"),
        )
    end
    return _discrete_symmetry_basis(
        basis;
        a,
        kblock,
        pblock,
        sblock,
        psblock,
    )
end

Base.length(basis::DiscreteBasis) = length(basis.encoded_states)
Base.:(==)(left::DiscreteBasis{K}, right::DiscreteBasis{K}) where {K} =
    left.L == right.L &&
    left.sps == right.sps &&
    left.conservation == right.conservation &&
    left.encoded_states == right.encoded_states &&
    left.symmetry.blocks == right.symmetry.blocks

function Base.getproperty(basis::DiscreteBasis{K}, name::Symbol) where {K}
    name in (:N, :L) && return getfield(basis, :L)
    name === :Ns && return length(getfield(basis, :encoded_states))
    name === :blocks && return copy(getfield(basis, :symmetry).blocks)
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
    ;
    sparse::Bool=false,
) where {T<:Number}
    return _full_projection_matrix(
        basis.encoded_states,
        basis.symmetry,
        basis.sps^basis.L,
        T,
        sparse_output=sparse,
    )
end

function project_from(
    basis::DiscreteBasis,
    vector::AbstractVecOrMat;
    sparse::Bool=true,
    pcon::Bool=false,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    return projection_matrix(
        basis,
        eltype(vector);
        sparse,
    ) * vector
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
    return_rdm=:both,
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
    if return_rdm in (:A, "A")
        return coefficients * coefficients'
    elseif return_rdm in (:B, "B")
        return coefficients' * coefficients
    elseif return_rdm in (:both, "both")
        return coefficients * coefficients', coefficients' * coefficients
    end
    throw(ArgumentError("return_rdm must be A, B, or both"))
end

function _discrete_reductions(
    basis::DiscreteBasis,
    state::AbstractMatrix,
    sites_A,
    return_rdm=:both,
)
    size(state) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("density matrix must match Ns"))
    sites_B = setdiff(collect(1:basis.L), sites_A)
    need_A = return_rdm in (:A, "A", :both, "both")
    need_B = return_rdm in (:B, "B", :both, "both")
    need_A || need_B ||
        throw(ArgumentError("return_rdm must be A, B, or both"))
    rho_A = need_A ?
        zeros(eltype(state), basis.sps^length(sites_A), basis.sps^length(sites_A)) :
        nothing
    rho_B = need_B ?
        zeros(eltype(state), basis.sps^length(sites_B), basis.sps^length(sites_B)) :
        nothing
    indices_A = Int[
        _local_index(@view(basis.occupations[row, :]), sites_A, basis.sps)
        for row in axes(basis.occupations, 1)
    ]
    indices_B = Int[
        _local_index(@view(basis.occupations[row, :]), sites_B, basis.sps)
        for row in axes(basis.occupations, 1)
    ]
    for row in axes(state, 1), column in axes(state, 2)
        row_A = indices_A[row]
        column_A = indices_A[column]
        row_B = indices_B[row]
        column_B = indices_B[column]
        need_A && row_B == column_B &&
            (rho_A[row_A, column_A] += state[row, column])
        need_B && row_A == column_A &&
            (rho_B[row_B, column_B] += state[row, column])
    end
    return need_A && need_B ? (rho_A, rho_B) :
           need_A ? rho_A : rho_B
end

function partial_trace(
    basis::DiscreteBasis,
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
        full_basis = if basis isa DiscreteBasis{:boson}
            BosonBasis1D(basis.L; sps=basis.sps)
        elseif basis isa DiscreteBasis{:spinless_fermion}
            SpinlessFermionBasis1D(basis.L)
        else
            SpinfulFermionBasis1D(basis.L)
        end
        return partial_trace(
            full_basis,
            expanded;
            sub_sys_A,
            return_rdm,
            enforce_pure,
            kwargs...,
        )
    end
    sites_A = _discrete_subsystem_sites(basis, sub_sys_A)
    return _discrete_reductions(
        basis,
        state,
        sites_A,
        return_rdm,
    )
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

function _spinful_mode_occupation(occupations, species::Symbol, site::Int)
    mask = species === :up ? 1 : 2
    return (occupations[site] & mask) == mask ? 1 : 0
end

function _fermion_prefix_occupation(
    basis::DiscreteBasis{:spinless_fermion},
    occupations,
    species::Symbol,
    site::Int,
)
    return site == 1 ? 0 : sum(@view occupations[1:(site - 1)])
end

function _fermion_prefix_occupation(
    basis::DiscreteBasis{:spinful_fermion},
    occupations,
    species::Symbol,
    site::Int,
)
    up_before = if species === :up
        site == 1 ? 0 : sum(
            _spinful_mode_occupation(occupations, :up, index)
            for index in 1:(site - 1)
        )
    else
        sum(
            _spinful_mode_occupation(occupations, :up, index)
            for index in 1:basis.L
        )
    end
    down_before = species === :down && site > 1 ?
        sum(
            _spinful_mode_occupation(occupations, :down, index)
            for index in 1:(site - 1)
        ) :
        0
    return up_before + down_before
end

function _apply_discrete_local(
    basis::DiscreteBasis{K},
    occupations,
    op,
    site,
    species::Symbol=:single,
) where {K}
    1 <= site <= basis.L ||
        throw(ArgumentError("site must lie in 1:$(basis.L)"))
    value = K === :spinful_fermion ?
        _spinful_mode_occupation(occupations, species, site) :
        occupations[site]
    if op == 'I'
        return one(Float64), true
    elseif op == 'n'
        return float(value), true
    elseif op == 'z'
        midpoint = K === :boson ? (basis.sps - 1) / 2 : 0.5
        return value - midpoint, true
    elseif op == '+'
        maximum_value = K === :boson ? basis.sps - 1 : 1
        value < maximum_value || return 0.0, false
        sign = K in (:spinless_fermion, :spinful_fermion) ?
            (-1)^_fermion_prefix_occupation(basis, occupations, species, site) :
            1
        if K === :spinful_fermion
            occupations[site] |= species === :up ? 1 : 2
        else
            occupations[site] += 1
        end
        factor = K === :boson ? sqrt(value + 1) : 1.0
        return sign * factor, true
    elseif op == '-'
        value > 0 || return 0.0, false
        sign = K in (:spinless_fermion, :spinful_fermion) ?
            (-1)^_fermion_prefix_occupation(basis, occupations, species, site) :
            1
        if K === :spinful_fermion
            occupations[site] &= ~(species === :up ? 1 : 2)
        else
            occupations[site] -= 1
        end
        factor = K === :boson ? sqrt(value) : 1.0
        return sign * factor, true
    end
    throw(ArgumentError("unsupported local operator '$op'"))
end

function _operator_actions(
    basis::DiscreteBasis{K},
    opstring::AbstractString,
    sites,
) where {K}
    if K === :spinful_fermion
        pieces = split(opstring, "|"; keepempty=true)
        length(pieces) == 2 ||
            throw(ArgumentError("spinful operator strings require one '|' separator"))
        expected = length(pieces[1]) + length(pieces[2])
        length(sites) == expected ||
            throw(ArgumentError("operator arity and sites differ"))
        actions = Tuple{Char,Int,Symbol}[]
        cursor = 1
        for op in pieces[1]
            push!(actions, (op, Int(sites[cursor]), :up))
            cursor += 1
        end
        for op in pieces[2]
            push!(actions, (op, Int(sites[cursor]), :down))
            cursor += 1
        end
        return actions
    end
    occursin('|', opstring) &&
        throw(ArgumentError("only spinful fermion operators use '|'"))
    length(sites) == length(opstring) ||
        throw(ArgumentError("operator arity and sites differ"))
    return [(op, Int(site), :single) for (op, site) in zip(opstring, sites)]
end

function operator_matrix(
    basis::DiscreteBasis{K},
    opstring::AbstractString,
    couplings,
    ;
    sparse::Bool=false,
) where {K}
    if _has_symmetry(basis.symmetry)
        parent_occupations = isempty(basis.symmetry.parent_states) ?
            Matrix{Int}(undef, 0, basis.L) :
            reduce(vcat, permutedims.(
                _digits(state, basis.L, basis.sps)
                for state in basis.symmetry.parent_states
            ))
        parent_symmetry = _identity_symmetry_data(
            basis.symmetry.parent_states,
            Dict(:conservation => basis.conservation),
        )
        parent = DiscreteBasis{K}(
            basis.L,
            basis.sps,
            basis.conservation,
            copy(basis.symmetry.parent_states),
            parent_occupations,
            copy(basis.symmetry.parent_lookup),
            basis.description,
            basis.operators,
            parent_symmetry,
        )
        parent_matrix = operator_matrix(
            parent,
            opstring,
            couplings;
            sparse=true,
        )
        projected =
            basis.symmetry.projector' *
            parent_matrix *
            basis.symmetry.projector
        return sparse ? projected : Matrix(projected)
    end
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    for coupling in couplings
        coefficient = first(coupling)
        sites = coupling[2:end]
        actions = _operator_actions(basis, opstring, sites)
        for column in axes(basis.occupations, 1)
            occupations = collect(@view basis.occupations[column, :])
            amplitude = complex(coefficient)
            alive = true
            # Operators act on kets from right to left. This is essential for
            # same-site products and for fermionic Jordan-Wigner signs.
            for (op, site, species) in Iterators.reverse(actions)
                factor, alive = _apply_discrete_local(
                    basis,
                    occupations,
                    op,
                    site,
                    species,
                )
                alive || break
                amplitude *= factor
            end
            alive || continue
            encoded = UInt64(sum(
                occupations[site] * basis.sps^(site - 1)
                for site in 1:basis.L
            ))
            row = get(basis.lookup, encoded, 0)
            row == 0 && continue
            push!(rows, row)
            push!(columns, column)
            push!(values, amplitude)
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

function inplace_op!(out, basis::DiscreteBasis, opstring, couplings)
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

function _parent_basis_for_checks(basis::SpinBasis1D)
    _has_symmetry(basis.symmetry) || return basis
    symmetry = _identity_symmetry_data(
        basis.symmetry.parent_states,
        Dict(:nup => basis.nup, :pauli => basis.pauli),
    )
    return SpinBasis1D(
        basis.L,
        basis.nup,
        basis.pauli,
        copy(basis.symmetry.parent_states),
        copy(basis.symmetry.parent_lookup),
        symmetry,
    )
end

function _parent_basis_for_checks(basis::DiscreteBasis{K}) where {K}
    _has_symmetry(basis.symmetry) || return basis
    occupations = isempty(basis.symmetry.parent_states) ?
        Matrix{Int}(undef, 0, basis.L) :
        reduce(vcat, permutedims.(
            _digits(state, basis.L, basis.sps)
            for state in basis.symmetry.parent_states
        ))
    symmetry = _identity_symmetry_data(
        basis.symmetry.parent_states,
        Dict(:conservation => basis.conservation),
    )
    return DiscreteBasis{K}(
        basis.L,
        basis.sps,
        basis.conservation,
        copy(basis.symmetry.parent_states),
        occupations,
        copy(basis.symmetry.parent_lookup),
        basis.description,
        basis.operators,
        symmetry,
    )
end

_parent_basis_for_checks(basis::AbstractBasis) = basis

function _entry_operator(entry)
    if hasproperty(entry, :op) && hasproperty(entry, :couplings)
        return String(getproperty(entry, :op)), getproperty(entry, :couplings)
    elseif (entry isa Tuple || entry isa AbstractVector) &&
           !isempty(entry) && first(entry) isa AbstractString
        return String(entry[1]), entry[2]
    end
    return nothing
end

function _entries_matrix(basis::AbstractBasis, entries)
    result = zeros(ComplexF64, length(basis), length(basis))
    for entry in entries
        if entry isa AbstractMatrix
            size(entry) == size(result) || return nothing
            result .+= entry
            continue
        end
        operator = _entry_operator(entry)
        operator === nothing && return nothing
        opstring, couplings = operator
        result .+= operator_matrix(basis, opstring, couplings)
    end
    return result
end

function _dynamic_groups(dynamic)
    groups = Vector{Vector{Any}}()
    keys = Any[]
    for entry in dynamic
        (entry isa Tuple || entry isa AbstractVector) || return nothing
        if first(entry) isa AbstractMatrix
            length(entry) == 3 || return nothing
            matrix, function_value, arguments = entry
            operator_entry = matrix
        else
            length(entry) == 4 || return nothing
            opstring, couplings, function_value, arguments = entry
            operator_entry = (String(opstring), couplings)
        end
        key = (function_value, Tuple(arguments))
        position = findfirst(existing -> isequal(existing, key), keys)
        if position === nothing
            push!(keys, key)
            push!(groups, Any[operator_entry])
        else
            push!(groups[position], operator_entry)
        end
    end
    return groups
end

function _operator_collections(basis, static, dynamic)
    parent = _parent_basis_for_checks(basis)
    static_matrix = _entries_matrix(parent, static)
    static_matrix === nothing && return nothing
    collections = Matrix{ComplexF64}[static_matrix]
    groups = _dynamic_groups(dynamic)
    groups === nothing && return nothing
    for group in groups
        matrix = _entries_matrix(parent, group)
        matrix === nothing && return nothing
        push!(collections, matrix)
    end
    return parent, collections
end

function check_hermitian(basis::AbstractBasis, static, dynamic=Any[])
    parsed = _operator_collections(basis, static, dynamic)
    parsed === nothing && return false
    _, matrices = parsed
    return all(matrices) do matrix
        residual = norm(matrix - matrix')
        residual <= 2e-10 * max(1.0, norm(matrix))
    end
end

function _particle_change(opstring::AbstractString)
    pieces = split(opstring, "|"; keepempty=true)
    change(piece) = count(==('+'), piece) - count(==('-'), piece)
    return Tuple(change(piece) for piece in pieces)
end

function _operator_strings(static, dynamic)
    strings = String[]
    for entry in static
        operator = _entry_operator(entry)
        operator === nothing && continue
        push!(strings, first(operator))
    end
    for entry in dynamic
        (entry isa Tuple || entry isa AbstractVector) || continue
        first(entry) isa AbstractString && push!(strings, String(first(entry)))
    end
    return strings
end

check_pcon(::AbstractBasis, static, dynamic=Any[]) = true

function check_pcon(basis::SpinBasis1D, static, dynamic=Any[])
    basis.nup === nothing && return true
    return all(_operator_strings(static, dynamic)) do opstring
        !any(character -> character in ('x', 'y'), opstring) &&
            _particle_change(opstring) == (0,)
    end
end

function check_pcon(basis::DiscreteBasis, static, dynamic=Any[])
    basis.conservation === nothing && return true
    return all(_operator_strings(static, dynamic)) do opstring
        changes = _particle_change(opstring)
        if basis isa DiscreteBasis{:spinful_fermion}
            length(changes) == 2 && changes == (0, 0)
        else
            length(changes) == 1 && changes == (0,)
        end
    end
end

check_symm(::AbstractBasis, static, dynamic=Any[]) = true

function _check_projected_symmetry(basis, static, dynamic)
    _has_symmetry(basis.symmetry) || return true
    parsed = _operator_collections(basis, static, dynamic)
    parsed === nothing && return false
    _, matrices = parsed
    projector = basis.symmetry.projector
    return all(matrices) do matrix
        action = matrix * projector
        residual = action - projector * (projector' * action)
        norm(residual) <= 3e-10 * max(1.0, norm(action))
    end
end

check_symm(basis::SpinBasis1D, static, dynamic=Any[]) =
    _check_projected_symmetry(basis, static, dynamic)
check_symm(basis::DiscreteBasis, static, dynamic=Any[]) =
    _check_projected_symmetry(basis, static, dynamic)

representative(basis::AbstractBasis, state::Integer) = state
normalization(basis::AbstractBasis, state::Integer) = one(Float64)
get_amp(basis::AbstractBasis, state::Integer) = one(Float64)

function _symmetry_column(data::SymmetryData, state::Integer)
    row = get(data.parent_lookup, UInt64(state), 0)
    row == 0 && throw(ArgumentError("state is outside the parent particle sector"))
    columns, values = findnz(data.projector[row, :])
    isempty(columns) &&
        throw(ArgumentError("state has zero weight in this symmetry sector"))
    position = argmax(abs.(values))
    return columns[position], values[position]
end

function representative(basis::Union{SpinBasis1D,DiscreteBasis}, state::Integer)
    _has_symmetry(basis.symmetry) || return state
    column, _ = _symmetry_column(basis.symmetry, state)
    return basis.encoded_states[column]
end

function normalization(basis::Union{SpinBasis1D,DiscreteBasis}, state::Integer)
    _has_symmetry(basis.symmetry) || return one(Float64)
    _, amplitude = _symmetry_column(basis.symmetry, state)
    return inv(abs2(amplitude))
end

function get_amp(basis::Union{SpinBasis1D,DiscreteBasis}, state::Integer)
    _has_symmetry(basis.symmetry) || return one(Float64)
    _, amplitude = _symmetry_column(basis.symmetry, state)
    return amplitude
end
make_basis!(basis::AbstractBasis) = basis
make_basis_blocks(basis::AbstractBasis) = [1:length(basis)]
project_to(basis::AbstractBasis, vector::AbstractVecOrMat) =
    projection_matrix(basis)' * vector
op_bra_ket(basis::DiscreteBasis, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)
op_shift_sector(target::AbstractBasis, source::AbstractBasis, operator, vector) =
    projection_matrix(target)' * projection_matrix(source) * vector
