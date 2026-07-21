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
const SpinlessFermionBasis1D = DiscreteBasis{:spinless_fermion}
const SpinfulFermionBasis1D = DiscreteBasis{:spinful_fermion}

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
    symmetry = _identity_symmetry_data(
        encoded,
        blocks,
        lookup;
        parent_occupations=occupations,
    )
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
    ;
    parallel::Bool=false,
) where {K}
    L > 0 || throw(ArgumentError("L must be positive"))
    sps >= 1 || throw(ArgumentError("sps must be positive"))
    dimension = BigInt(sps)^L
    dimension <= typemax(UInt64) ||
        throw(ArgumentError("basis encoding exceeds UInt64"))
    encoded = UInt64[]
    if parallel && Threads.nthreads() > 1
        dimension <= typemax(Int) ||
            throw(ArgumentError(
                "parallel basis construction requires an Int-sized state space",
            ))
        # `threadid()` may include the interactive pool, so allocate by the
        # maximum runtime id rather than only the default-pool thread count.
        local_states = [UInt64[] for _ in 1:Threads.maxthreadid()]
        Threads.@threads for raw_value in 0:(Int(dimension) - 1)
            value = UInt64(raw_value)
            occupations = _digits(value, L, sps)
            keep(occupations) || continue
            push!(local_states[Threads.threadid()], value)
        end
        encoded = reduce(vcat, local_states; init=UInt64[])
        sort!(encoded)
    else
        for value in UInt64(0):(UInt64(dimension) - 1)
            occupations = _digits(value, L, sps)
            keep(occupations) || continue
            push!(encoded, value)
        end
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
    projector = nothing
    blocks = copy(basis.symmetry.blocks)
    order = basis.L ÷ Int(a)
    translation = [mod1(site + Int(a), basis.L) for site in 1:basis.L]
    parity = [basis.L - site + 1 for site in 1:basis.L]
    fermionic = K in (:spinless_fermion, :spinful_fermion)
    spinful = K === :spinful_fermion
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.L
    ]

    function site_transform(permutation)
        return state -> _site_permutation_transform(
            state,
            basis.sps,
            permutation,
            weights;
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
        encoded = zero(UInt64)
        for site in 1:basis.L
            occupation = Int((state ÷ weights[site]) % UInt64(basis.sps))
            encoded += UInt64(basis.sps - 1 - occupation) * weights[site]
        end
        encoded, 1.0 + 0im
    end
    cA_transform = state -> begin
        encoded = state
        for site in 1:2:basis.L
            occupation = Int((state ÷ weights[site]) % UInt64(basis.sps))
            encoded = UInt64(
                Int128(encoded) +
                Int128(basis.sps - 1 - 2 * occupation) * Int128(weights[site]),
            )
        end
        encoded, 1.0 + 0im
    end
    cB_transform = state -> begin
        encoded = state
        for site in 2:2:basis.L
            occupation = Int((state ÷ weights[site]) % UInt64(basis.sps))
            encoded = UInt64(
                Int128(encoded) +
                Int128(basis.sps - 1 - 2 * occupation) * Int128(weights[site]),
            )
        end
        encoded, 1.0 + 0im
    end
    spin_swap_transform = state -> begin
        K === :spinful_fermion ||
            throw(ArgumentError("sblock is defined only for spinful fermions"))
        encoded = zero(UInt64)
        occupied_up = 0
        occupied_down = 0
        for site in 1:basis.L
            digit = Int((state ÷ weights[site]) % UInt64(basis.sps))
            transformed = ((digit & 1) << 1) | ((digit & 2) >> 1)
            encoded += UInt64(transformed) * weights[site]
            occupied_up += digit & 1
            occupied_down += (digit & 2) >> 1
        end
        inversions = occupied_up * occupied_down
        phase = iseven(inversions) ? 1.0 + 0im : -1.0 + 0im
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
        projector = if projector === nothing
            _cyclic_projector(
                parent_states,
                lookup,
                transform,
                ComplexF64(value),
            )
        else
            symmetry_matrix =
                _signed_permutation(parent_states, lookup, transform)
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
    symmetry = _finalize_symmetry_data(
        parent_states,
        projector,
        blocks,
    )
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

"""
    HOBasis(Np)

Truncated single-mode harmonic-oscillator basis with number states
`0:Np`. This is the Julia counterpart of QuSpin's public `ho_basis`.
"""
function HOBasis(Np::Integer)
    Np >= 0 || throw(ArgumentError("Np must be nonnegative"))
    return BosonBasis1D(1; sps=Int(Np) + 1)
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
            ("I", "+", "-", "n", "z", "x", "y"),
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
            ("I", "+", "-", "n", "z", "x", "y"),
        )
    end
    return _discrete_symmetry_basis(basis; a, kblock, pblock)
end

function DiscreteBasis{:spinful_fermion}(
    L::Integer;
    Nf=nothing,
    nf=nothing,
    double_occupancy::Bool=true,
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
    sectors = if selected_particles === nothing
        nothing
    elseif selected_particles isa Tuple &&
           length(selected_particles) == 2 &&
           all(value -> value isa Integer, selected_particles)
        Tuple{Int,Int}[(Int(selected_particles[1]), Int(selected_particles[2]))]
    else
        Tuple{Int,Int}[
            (Int(sector[1]), Int(sector[2]))
            for sector in selected_particles
        ]
    end
    keep = occupations -> begin
        double_occupancy || all(digit -> digit != 3, occupations) || return false
        sectors === nothing && return true
        particle_numbers = (
            count(digit -> digit & 1 == 1, occupations),
            count(digit -> digit & 2 == 2, occupations),
        )
        return particle_numbers in sectors
    end
    basis = if sectors === nothing
        _make_discrete_basis(
            Val(:spinful_fermion),
            L,
            4,
            selected_particles,
            keep,
            "spinful fermion lattice basis",
            ("I", "+", "-", "n", "z", "x", "y", "|"),
        )
    elseif all(
        sector -> all(particles -> 0 <= particles <= L, sector),
        sectors,
    )
        encoded = UInt64[]
        for (up_particles, down_particles) in sectors
            up_states = _fixed_weight_states(Int(L), up_particles)
            down_states = _fixed_weight_states(Int(L), down_particles)
            for up in up_states, down in down_states
                value = zero(UInt64)
                valid = true
                for site in 0:(Int(L) - 1)
                    up_bit = (up >> site) & UInt64(1)
                    down_bit = (down >> site) & UInt64(1)
                    if !double_occupancy && isone(up_bit) && isone(down_bit)
                        valid = false
                        break
                    end
                    value |= (up_bit | (down_bit << 1)) << (2 * site)
                end
                valid && push!(encoded, value)
            end
        end
        sort!(unique!(encoded))
        _discrete_basis_from_encoded(
            Val(:spinful_fermion),
            L,
            4,
            selected_particles,
            encoded,
            "spinful fermion lattice basis",
            ("I", "+", "-", "n", "z", "x", "y", "|"),
        )
    else
        _discrete_basis_from_encoded(
            Val(:spinful_fermion),
            L,
            4,
            selected_particles,
            UInt64[],
            "spinful fermion lattice basis",
            ("I", "+", "-", "n", "z", "x", "y", "|"),
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

function DiscreteBasis{:spin}(
    L::Integer;
    S=1,
    nup=nothing,
    Nup=nothing,
    m=nothing,
    pauli::Bool=false,
    a::Integer=1,
    kblock=nothing,
    pblock=nothing,
    zblock=nothing,
    pzblock=nothing,
    zAblock=nothing,
    zBblock=nothing,
)
    spin = _parse_spin_value(S)
    sps = Int(2spin + 1)
    nup !== nothing && Nup !== nothing &&
        throw(ArgumentError("specify only one of nup and Nup"))
    selected = nup === nothing ? Nup : nup
    if m !== nothing
        selected === nothing ||
            throw(ArgumentError("m cannot be combined with nup or Nup"))
        selected = round(Int, (float(m) + float(spin)) * L)
    end
    totals = selected === nothing ?
        nothing :
        Set(selected isa Integer ? [Int(selected)] : Int.(collect(selected)))
    totals === nothing || all(0 .<= collect(totals) .<= L * (sps - 1)) ||
        throw(ArgumentError("Nup lies outside the higher-spin Hilbert space"))
    encoded = totals === nothing ?
        collect(UInt64(0):(UInt64(sps)^Int(L) - UInt64(1))) :
        _bounded_composition_states(Int(L), sps, totals)
    basis = _discrete_basis_from_encoded(
        Val(:spin),
        L,
        sps,
        selected,
        encoded,
        "spin-$spin lattice basis",
        ("I", "+", "-", "z", "x", "y"),
    )
    any(value !== nothing for value in (zblock, pzblock, zAblock, zBblock)) &&
        throw(ArgumentError(
            "higher-spin inversion blocks are available through SpinBasisGeneral maps",
        ))
    return _discrete_symmetry_basis(basis; a, kblock, pblock)
end

function _advanced_fermion_transform(
    basis::DiscreteBasis{K},
    map,
) where {K}
    values = Int.(collect(map))
    mode_count = K === :spinful_fermion ? 2basis.L : basis.L
    length(values) == mode_count ||
        throw(ArgumentError(
            "advanced fermion maps require one entry per fermionic mode",
        ))
    zero_based = any(iszero, values) || any(<(0), values)
    destinations = Vector{Int}(undef, mode_count)
    flips = falses(mode_count)
    for source in 1:mode_count
        value = values[source]
        destinations[source] = value < 0 ?
            -value :
            zero_based ? value + 1 : value
        flips[source] = value < 0
    end
    sort(destinations) == collect(1:mode_count) ||
        throw(ArgumentError(
            "advanced fermion maps must be signed mode permutations",
        ))
    order = _general_spin_map_order(destinations, flips)
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.L
    ]
    transform = state -> begin
        source_occupied = falses(mode_count)
        if K === :spinless_fermion
            for mode in 1:mode_count
                source_occupied[mode] =
                    !iszero(state & (UInt64(1) << (mode - 1)))
            end
        else
            for site in 1:basis.L
                digit = Int((state ÷ weights[site]) % UInt64(4))
                source_occupied[site] = digit & 1 == 1
                source_occupied[basis.L + site] = digit & 2 == 2
            end
        end

        target_occupied = falses(mode_count)
        for source in 1:mode_count
            flips[source] &&
                (target_occupied[destinations[source]] = true)
        end
        phase = 1.0 + 0im
        for source in mode_count:-1:1
            source_occupied[source] || continue
            destination = destinations[source]
            parity = count(@view target_occupied[1:(destination - 1)])
            isodd(parity) && (phase = -phase)
            if flips[source]
                target_occupied[destination] ||
                    throw(ArgumentError(
                        "invalid particle-hole canonical transformation",
                    ))
                target_occupied[destination] = false
                isodd(destination - 1) && (phase = -phase)
            else
                !target_occupied[destination] ||
                    throw(ArgumentError(
                        "invalid fermion mode permutation",
                    ))
                target_occupied[destination] = true
            end
        end

        encoded = zero(UInt64)
        if K === :spinless_fermion
            for mode in 1:mode_count
                target_occupied[mode] &&
                    (encoded |= UInt64(1) << (mode - 1))
            end
        else
            for site in 1:basis.L
                target_occupied[site] &&
                    (encoded += weights[site])
                target_occupied[basis.L + site] &&
                    (encoded += UInt64(2) * weights[site])
            end
        end
        return encoded, phase
    end
    return transform, order
end

function _general_discrete_transform(
    basis::DiscreteBasis{K},
    map,
) where {K}
    if K === :spinful_fermion && length(map) == 2basis.L
        return _advanced_fermion_transform(basis, map)
    elseif K === :spinless_fermion &&
           length(map) == basis.L &&
           any(<(0), map)
        return _advanced_fermion_transform(basis, map)
    end
    permutation, flips =
        _normalize_general_spin_map(map, basis.L)
    order = _general_spin_map_order(permutation, flips)
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.L
    ]
    if K === :spinful_fermion && any(flips)
        mode_permutation = Vector{Int}(undef, 2basis.L)
        for source in 1:basis.L
            destination = permutation[source]
            if flips[source]
                mode_permutation[source] = basis.L + destination
                mode_permutation[basis.L + source] = destination
            else
                mode_permutation[source] = destination
                mode_permutation[basis.L + source] = basis.L + destination
            end
        end
        transform = state -> begin
            occupied_modes = Int[]
            encoded = zero(UInt64)
            for source in 1:basis.L
                digit = Int((state ÷ weights[source]) % UInt64(4))
                digit & 1 == 1 && push!(occupied_modes, source)
                digit & 2 == 2 && push!(occupied_modes, basis.L + source)
            end
            sort!(occupied_modes)
            for mode in occupied_modes
                destination_mode = mode_permutation[mode]
                destination_site = mod1(destination_mode, basis.L)
                bit = destination_mode <= basis.L ? UInt64(1) : UInt64(2)
                encoded += bit * weights[destination_site]
            end
            phase = _permutation_phase(occupied_modes, mode_permutation)
            return encoded, phase
        end
        return transform, order
    elseif K in (:spinless_fermion, :spinful_fermion) && any(flips)
        throw(ArgumentError(
            "particle-hole maps require advanced fermion notation",
        ))
    elseif K === :boson && any(flips)
        transform = state -> begin
            encoded = zero(UInt64)
            for source in 1:basis.L
                occupation =
                    Int((state ÷ weights[source]) % UInt64(basis.sps))
                flips[source] &&
                    (occupation = basis.sps - 1 - occupation)
                encoded += UInt64(occupation) * weights[permutation[source]]
            end
            return encoded, 1.0 + 0im
        end
        return transform, order
    end
    fermionic = K in (:spinless_fermion, :spinful_fermion)
    spinful = K === :spinful_fermion
    transform = state -> _site_permutation_transform(
        state,
        basis.sps,
        permutation,
        weights;
        fermionic,
        spinful,
    )
    return transform, order
end

function _general_discrete_basis(
    basis::DiscreteBasis{K},
    block_order,
    blocks,
) where {K}
    isempty(blocks) && return basis
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
    symmetry_blocks = copy(basis.symmetry.blocks)
    for name in ordered_names
        specification = block_dictionary[name]
        (specification isa Tuple || specification isa AbstractVector) &&
            length(specification) >= 2 ||
            throw(ArgumentError(
                "general symmetry blocks must be (map, quantum_number)",
            ))
        transform, order =
            _general_discrete_transform(basis, specification[1])
        quantum_number = Int(specification[2])
        eigenvalue = cis(2π * mod(quantum_number, order) / order)
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
        symmetry_blocks[name] = mod(quantum_number, order)
        symmetry_blocks[Symbol(name, :_period)] = order
    end
    symmetry_blocks[:block_order] = copy(ordered_names)
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

function _deferred_discrete_metadata(
    L,
    sps,
    conservation,
    description,
    operators,
    blocks,
    Ns_block_est,
)
    metadata = Dict{Symbol,Any}(
        :L => Int(L),
        :N => Int(L),
        :Ns => 1,
        :sps => Int(sps),
        :dtype => UInt64,
        :states => UInt64[0],
        :encoded_states => UInt64[0],
        :conservation => conservation,
        :description => description,
        :operators => Tuple(operators),
        :noncommuting_bits => Tuple{Vector{Int},Int}[],
        :blocks => Dict{Symbol,Any}(
            Symbol(name) => value for (name, value) in blocks
        ),
    )
    Ns_block_est === nothing ||
        (metadata[:Ns_block_est] = Int(Ns_block_est))
    return metadata
end

function BosonBasisGeneral(
    L::Integer;
    Nb=nothing,
    nb=nothing,
    sps::Union{Nothing,Integer}=nothing,
    Ns_block_est=nothing,
    make_basis::Bool=true,
    block_order=nothing,
    blocks...,
)
    if !make_basis
        block_keywords = (; blocks...)
        selected_particles = Nb === nothing ? nb : Nb
        local_dimension = sps === nothing ?
            (
                selected_particles isa Integer ?
                Int(selected_particles) + 1 :
                2
            ) :
            Int(sps)
        builder = () -> BosonBasisGeneral(
            L;
            Nb,
            nb,
            sps,
            Ns_block_est,
            make_basis=true,
            block_order,
            block_keywords...,
        )
        metadata = _deferred_discrete_metadata(
            L,
            local_dimension,
            selected_particles,
            "deferred boson general basis",
            ("I", "+", "-", "n", "z"),
            blocks,
            Ns_block_est,
        )
        return _deferred_basis(DiscreteBasis{:boson}, builder, metadata)
    end
    basis = BosonBasis1D(L; Nb, nb, sps)
    return _general_discrete_basis(basis, block_order, blocks)
end

function SpinlessFermionBasisGeneral(
    L::Integer;
    Nf=nothing,
    nf=nothing,
    Ns_block_est=nothing,
    make_basis::Bool=true,
    block_order=nothing,
    blocks...,
)
    if !make_basis
        block_keywords = (; blocks...)
        selected_particles = Nf === nothing ? nf : Nf
        builder = () -> SpinlessFermionBasisGeneral(
            L;
            Nf,
            nf,
            Ns_block_est,
            make_basis=true,
            block_order,
            block_keywords...,
        )
        metadata = _deferred_discrete_metadata(
            L,
            2,
            selected_particles,
            "deferred spinless-fermion general basis",
            ("I", "+", "-", "n", "z"),
            blocks,
            Ns_block_est,
        )
        metadata[:noncommuting_bits] = [collect(1:Int(L)) => -1]
        return _deferred_basis(
            DiscreteBasis{:spinless_fermion},
            builder,
            metadata,
        )
    end
    basis = SpinlessFermionBasis1D(L; Nf, nf)
    return _general_discrete_basis(basis, block_order, blocks)
end

function SpinfulFermionBasisGeneral(
    L::Integer;
    Nf=nothing,
    nf=nothing,
    Ns_block_est=nothing,
    simple_symm::Bool=true,
    make_basis::Bool=true,
    block_order=nothing,
    double_occupancy::Bool=true,
    blocks...,
)
    if !make_basis
        block_keywords = (; blocks...)
        selected_particles = Nf === nothing ? nf : Nf
        builder = () -> SpinfulFermionBasisGeneral(
            L;
            Nf,
            nf,
            Ns_block_est,
            simple_symm,
            make_basis=true,
            block_order,
            double_occupancy,
            block_keywords...,
        )
        metadata = _deferred_discrete_metadata(
            2Int(L),
            2,
            selected_particles,
            "deferred spinful-fermion general basis",
            ("I", "+", "-", "n", "z", "|"),
            blocks,
            Ns_block_est,
        )
        metadata[:L] = Int(L)
        metadata[:N] = Int(L)
        metadata[:double_occupancy] = double_occupancy
        return _deferred_basis(
            DiscreteBasis{:spinful_fermion},
            builder,
            metadata,
        )
    end
    basis = SpinfulFermionBasis1D(L; Nf, nf, double_occupancy)
    return _general_discrete_basis(basis, block_order, blocks)
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
    name === :Np && K === :boson && getfield(basis, :L) == 1 &&
        return getfield(basis, :sps) - 1
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
    pcon::Bool=false,
) where {T<:Number}
    if pcon
        projected = _parent_projection_matrix(basis.symmetry, T)
        return sparse ? projected : Matrix(projected)
    end
    return _full_projection_matrix(
        basis.encoded_states,
        basis.symmetry,
        basis.sps^basis.L,
        T,
        sparse_output=sparse,
    )
end

_full_projection_dimension(basis::DiscreteBasis) = basis.sps^basis.L
_projection_output_dimension(basis::DiscreteBasis, pcon::Bool) =
    pcon ? length(basis.symmetry.parent_states) :
    _full_projection_dimension(basis)
_pcon_projection_matrix(basis::DiscreteBasis, ::Type{T}) where {T<:Number} =
    _parent_projection_matrix(basis.symmetry, T)

function project_from(
    basis::DiscreteBasis,
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
        basis.sps^basis.L,
        sparse,
    )
end

get_vec(basis::DiscreteBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)

function _xy_expansion_choices(
    ::DiscreteBasis{K},
    operator::Char,
) where {K}
    if K === :spin
        operator == 'x' &&
            return (('+', 0.5 + 0im), ('-', 0.5 + 0im))
        operator == 'y' &&
            return (('+', -0.5im), ('-', 0.5im))
    elseif K in (:spinless_fermion, :spinful_fermion)
        operator == 'x' &&
            return (('+', 1.0 + 0im), ('-', 1.0 + 0im))
        operator == 'y' &&
            return (('+', 1.0im), ('-', -1.0im))
    end
    return ((operator, 1.0 + 0im),)
end

expanded_form(basis::DiscreteBasis, static=Any[], dynamic=Any[]) = (
    _expanded_form_entries(basis, static, false),
    _expanded_form_entries(basis, dynamic, true),
)

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

function _discrete_parent_basis(basis::DiscreteBasis{K}) where {K}
    parent_states = basis.symmetry.parent_states
    occupation_cache = basis.symmetry.parent_occupations
    parent_occupations = occupation_cache[]
    if parent_occupations === nothing
        parent_occupations = Matrix{Int}(
            undef,
            length(parent_states),
            basis.L,
        )
        for (row, state) in pairs(parent_states)
            remaining = state
            for site in 1:basis.L
                parent_occupations[row, site] =
                    Int(rem(remaining, UInt64(basis.sps)))
                remaining = div(remaining, UInt64(basis.sps))
            end
        end
        occupation_cache[] = parent_occupations
    end
    parent_lookup = basis.symmetry.parent_lookup
    parent_symmetry = _identity_symmetry_data(
        parent_states,
        Dict(:conservation => basis.conservation),
        parent_lookup,
        parent_occupations=parent_occupations,
    )
    return DiscreteBasis{K}(
        basis.L,
        basis.sps,
        basis.conservation,
        parent_states,
        parent_occupations,
        parent_lookup,
        basis.description,
        basis.operators,
        parent_symmetry,
    )
end

function _discrete_pure_coefficients(
    basis::DiscreteBasis,
    state::AbstractVector,
    sites_A,
)
    parent_basis = basis
    parent_state = state
    if _has_symmetry(basis.symmetry)
        parent_basis = _discrete_parent_basis(basis)
        parent_state = basis.symmetry.projector * state
    end
    length(parent_state) == length(parent_basis) ||
        throw(DimensionMismatch("state length must equal Ns"))
    sites_B = setdiff(collect(1:parent_basis.L), sites_A)
    coefficients = zeros(
        eltype(parent_state),
        parent_basis.sps^length(sites_A),
        parent_basis.sps^length(sites_B),
    )
    for row in axes(parent_basis.occupations, 1)
        index_A = _local_index(
            @view(parent_basis.occupations[row, :]),
            sites_A,
            parent_basis.sps,
        )
        index_B = _local_index(
            @view(parent_basis.occupations[row, :]),
            sites_B,
            parent_basis.sps,
        )
        coefficients[index_A, index_B] = parent_state[row]
    end
    return coefficients
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
    if need_A
        groups_B = Dict{Int,Vector{Int}}()
        for position in eachindex(indices_B)
            push!(get!(Vector{Int}, groups_B, indices_B[position]), position)
        end
        for group in values(groups_B), row in group, column in group
            rho_A[indices_A[row], indices_A[column]] += state[row, column]
        end
    end
    if need_B
        groups_A = Dict{Int,Vector{Int}}()
        for position in eachindex(indices_A)
            push!(get!(Vector{Int}, groups_A, indices_A[position]), position)
        end
        for group in values(groups_A), row in group, column in group
            rho_B[indices_B[row], indices_B[column]] += state[row, column]
        end
    end
    return need_A && need_B ? (rho_A, rho_B) :
           need_A ? rho_A : rho_B
end

function _spinful_species_reductions(
    basis::DiscreteBasis{:spinful_fermion},
    state::AbstractVecOrMat,
    subsystem,
    return_rdm,
)
    length(subsystem) == 2 ||
        throw(ArgumentError("spinful subsystem must be (up_sites, down_sites)"))
    up_sites = sort!(unique(Int.(collect(subsystem[1]))))
    down_sites = sort!(unique(Int.(collect(subsystem[2]))))
    all(site -> 1 <= site <= basis.L, up_sites) &&
        all(site -> 1 <= site <= basis.L, down_sites) ||
        throw(ArgumentError("spinful subsystem sites lie outside the lattice"))
    selected_modes = vcat(up_sites, basis.L .+ down_sites)
    complement_modes =
        setdiff(collect(1:(2basis.L)), selected_modes)
    dimension_A = 1 << length(selected_modes)
    dimension_B = 1 << length(complement_modes)
    function mode_index(occupations, modes)
        index = 0
        for mode in modes
            site = mod1(mode, basis.L)
            digit = occupations[site]
            occupied = mode <= basis.L ?
                digit & 1 == 1 :
                digit & 2 == 2
            index = (index << 1) | Int(occupied)
        end
        return index + 1
    end
    indices_A = [
        mode_index(@view(basis.occupations[row, :]), selected_modes)
        for row in axes(basis.occupations, 1)
    ]
    indices_B = [
        mode_index(@view(basis.occupations[row, :]), complement_modes)
        for row in axes(basis.occupations, 1)
    ]
    if state isa AbstractVector
        length(state) == length(basis) ||
            throw(DimensionMismatch("state length must equal Ns"))
        coefficients =
            zeros(eltype(state), dimension_A, dimension_B)
        for row in eachindex(state)
            coefficients[indices_A[row], indices_B[row]] = state[row]
        end
        rho_A = coefficients * coefficients'
        rho_B = coefficients' * coefficients
    else
        size(state) == (length(basis), length(basis)) ||
            throw(DimensionMismatch("density matrix must match Ns"))
        rho_A = zeros(eltype(state), dimension_A, dimension_A)
        rho_B = zeros(eltype(state), dimension_B, dimension_B)
        for row in axes(state, 1), column in axes(state, 2)
            indices_B[row] == indices_B[column] &&
                (rho_A[indices_A[row], indices_A[column]] += state[row, column])
            indices_A[row] == indices_A[column] &&
                (rho_B[indices_B[row], indices_B[column]] += state[row, column])
        end
    end
    return return_rdm in (:A, "A") ? rho_A :
           return_rdm in (:B, "B") ? rho_B :
           return_rdm in (:both, "both") ? (rho_A, rho_B) :
           throw(ArgumentError("return_rdm must be A, B, or both"))
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
        projector = basis.symmetry.projector
        expanded = if state isa AbstractMatrix &&
                      size(state) == (length(basis), length(basis)) &&
                      !enforce_pure
            projector * state * projector'
        else
            projector * state
        end
        return partial_trace(
            _discrete_parent_basis(basis),
            expanded;
            sub_sys_A,
            return_rdm,
            enforce_pure,
            kwargs...,
        )
    end
    if basis isa DiscreteBasis{:spinful_fermion} &&
       sub_sys_A isa Tuple
        return _spinful_species_reductions(
            basis,
            state,
            sub_sys_A,
            return_rdm,
        )
    end
    sites_A = _discrete_subsystem_sites(basis, sub_sys_A)
    if state isa AbstractMatrix && enforce_pure &&
       size(state, 1) == length(basis)
        reductions = [
            _discrete_reductions(
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
    enforce_pure::Bool=false,
    return_rdm_EVs::Bool=false,
    alpha::Real=1.0,
    kwargs...,
)
    sites_A = _discrete_subsystem_sites(basis, sub_sys_A)
    if state isa AbstractVector
        coefficients = _discrete_pure_coefficients(basis, state, sites_A)
        probabilities = _schmidt_probabilities(coefficients)
        entropy = _entropy_from_probabilities(probabilities, alpha)
        normalization_A = density && !isempty(sites_A) ? length(sites_A) : 1
        sites_B = basis.L - length(sites_A)
        normalization_B = density && sites_B > 0 ? sites_B : 1
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
        return_rdm_EVs && (result["p_A"] = collect(probabilities))
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
    result = Dict{String,Any}(
        "Sent_A" => _entropy_from_probabilities(
            probabilities_A,
            alpha,
        ) / normalization_A,
    )
    if return_rdm in (:A, "A", :both, "both")
        result["rdm_A"] = rho_A
    end
    if need_B
        result["Sent_B"] = _entropy_from_density(rho_B, alpha) / normalization_B
        result["rdm_B"] = rho_B
    end
    return_rdm_EVs && (result["p_A"] = probabilities_A)
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
        midpoint = K in (:boson, :spin) ? (basis.sps - 1) / 2 : 0.5
        return value - midpoint, true
    elseif op == '+'
        maximum_value = K in (:boson, :spin) ? basis.sps - 1 : 1
        value < maximum_value || return 0.0, false
        prefix = K in (:spinless_fermion, :spinful_fermion) ?
            _fermion_prefix_occupation(basis, occupations, species, site) :
            0
        sign = isodd(prefix) ? -1 : 1
        if K === :spinful_fermion
            occupations[site] |= species === :up ? 1 : 2
        else
            occupations[site] += 1
        end
        factor = if K === :boson
            sqrt(value + 1)
        elseif K === :spin
            spin = (basis.sps - 1) / 2
            magnetic = value - spin
            sqrt(spin * (spin + 1) - magnetic * (magnetic + 1))
        else
            1.0
        end
        return sign * factor, true
    elseif op == '-'
        value > 0 || return 0.0, false
        prefix = K in (:spinless_fermion, :spinful_fermion) ?
            _fermion_prefix_occupation(basis, occupations, species, site) :
            0
        sign = isodd(prefix) ? -1 : 1
        if K === :spinful_fermion
            occupations[site] &= ~(species === :up ? 1 : 2)
        else
            occupations[site] -= 1
        end
        factor = if K === :boson
            sqrt(value)
        elseif K === :spin
            spin = (basis.sps - 1) / 2
            magnetic = value - spin
            sqrt(spin * (spin + 1) - magnetic * (magnetic - 1))
        else
            1.0
        end
        return sign * factor, true
    end
    throw(ArgumentError("unsupported local operator '$op'"))
end

@inline function _encoded_occupation(
    encoded::UInt64,
    weight::UInt64,
    sps::Int,
)
    return Int((encoded ÷ weight) % UInt64(sps))
end

@inline function _spinful_encoded_occupation(
    encoded::UInt64,
    weight::UInt64,
    species::Symbol,
)
    digit = Int((encoded ÷ weight) % UInt64(4))
    mask = species === :up ? 1 : 2
    return (digit & mask) == mask ? 1 : 0
end

@inline function _fermion_prefix_encoded(
    basis::DiscreteBasis{:spinless_fermion},
    encoded::UInt64,
    species::Symbol,
    site::Int,
    weights,
)
    site == 1 && return 0
    return count_ones(encoded & (weights[site] - UInt64(1)))
end

@inline function _fermion_prefix_encoded(
    basis::DiscreteBasis{:spinful_fermion},
    encoded::UInt64,
    species::Symbol,
    site::Int,
    weights,
)
    # A radix-four spinful state stores the up/down occupations as the
    # even/odd bits of each two-bit site digit.  Count the Jordan-Wigner
    # prefix directly instead of scanning all sites for every local action.
    lower_site_bits = (UInt64(1) << (2 * (site - 1))) - UInt64(1)
    up_bits = encoded & UInt64(0x5555555555555555)
    species === :up && return count_ones(up_bits & lower_site_bits)
    down_bits = encoded & UInt64(0xaaaaaaaaaaaaaaaa)
    return count_ones(up_bits) + count_ones(down_bits & lower_site_bits)
end

"""
Apply one local discrete operator directly to the radix-encoded state.

This is the allocation-free action kernel used by sparse assembly and the
matrix-free linear operator. It keeps the encoded state authoritative, so a
column never needs an occupation-vector copy or a full re-encoding pass.
"""
@inline function _apply_discrete_encoded_local(
    basis::DiscreteBasis{K},
    encoded::UInt64,
    op::Char,
    site::Int,
    species::Symbol,
    weight::UInt64,
    weights,
) where {K}
    1 <= site <= basis.L ||
        throw(ArgumentError("site must lie in 1:$(basis.L)"))
    value = K === :spinful_fermion ?
        _spinful_encoded_occupation(encoded, weight, species) :
        _encoded_occupation(encoded, weight, basis.sps)
    op == 'I' && return encoded, one(Float64), true
    op == 'n' && return encoded, float(value), true
    if op == 'z'
        midpoint = K in (:boson, :spin) ? (basis.sps - 1) / 2 : 0.5
        return encoded, value - midpoint, true
    end
    if op == '+'
        maximum_value = K in (:boson, :spin) ? basis.sps - 1 : 1
        value < maximum_value || return encoded, 0.0, false
        prefix = K in (:spinless_fermion, :spinful_fermion) ?
            _fermion_prefix_encoded(
                basis,
                encoded,
                species,
                site,
                weights,
            ) :
            0
        sign = isodd(prefix) ? -1 : 1
        increment = K === :spinful_fermion ?
            UInt64(species === :up ? 1 : 2) * weight :
            weight
        factor = if K === :boson
            sqrt(value + 1)
        elseif K === :spin
            spin = (basis.sps - 1) / 2
            magnetic = value - spin
            sqrt(spin * (spin + 1) - magnetic * (magnetic + 1))
        else
            1.0
        end
        return encoded + increment, sign * factor, true
    end
    if op == '-'
        value > 0 || return encoded, 0.0, false
        prefix = K in (:spinless_fermion, :spinful_fermion) ?
            _fermion_prefix_encoded(
                basis,
                encoded,
                species,
                site,
                weights,
            ) :
            0
        sign = isodd(prefix) ? -1 : 1
        decrement = K === :spinful_fermion ?
            UInt64(species === :up ? 1 : 2) * weight :
            weight
        factor = if K === :boson
            sqrt(value)
        elseif K === :spin
            spin = (basis.sps - 1) / 2
            magnetic = value - spin
            sqrt(spin * (spin + 1) - magnetic * (magnetic - 1))
        else
            1.0
        end
        return encoded - decrement, sign * factor, true
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

function _discrete_operator_triplets(
    basis::DiscreteBasis,
    opstring::AbstractString,
    couplings,
)
    if basis isa Union{
        DiscreteBasis{:spinless_fermion},
        DiscreteBasis{:spinful_fermion},
    } && any(operator -> operator in ('x', 'y'), opstring)
        expansions = Tuple{String,ComplexF64}[("", 1.0 + 0im)]
        for operator in opstring
            choices = if operator == 'x'
                (('+', 1.0 + 0im), ('-', 1.0 + 0im))
            elseif operator == 'y'
                (('+', 1.0im), ('-', -1.0im))
            else
                ((operator, 1.0 + 0im),)
            end
            expansions = [
                (prefix * string(expanded), coefficient * scale)
                for (prefix, coefficient) in expansions
                for (expanded, scale) in choices
            ]
        end
        rows = Int[]
        columns = Int[]
        values = ComplexF64[]
        for (expanded, scale) in expansions
            scaled_couplings = [
                (scale * first(coupling), Base.tail(coupling)...)
                for coupling in couplings
            ]
            expanded_rows, expanded_columns, expanded_values =
                _discrete_operator_triplets(
                    basis,
                    expanded,
                    scaled_couplings,
                )
            append!(rows, expanded_rows)
            append!(columns, expanded_columns)
            append!(values, expanded_values)
        end
        return rows, columns, values
    end
    if basis isa DiscreteBasis{:spin} &&
       any(operator -> operator in ('x', 'y'), opstring)
        expansions = Tuple{String,ComplexF64}[("", 1.0 + 0im)]
        for operator in opstring
            choices = if operator == 'x'
                (('+', 0.5 + 0im), ('-', 0.5 + 0im))
            elseif operator == 'y'
                (('+', -0.5im), ('-', 0.5im))
            else
                ((operator, 1.0 + 0im),)
            end
            expansions = [
                (prefix * string(expanded), coefficient * scale)
                for (prefix, coefficient) in expansions
                for (expanded, scale) in choices
            ]
        end
        rows = Int[]
        columns = Int[]
        values = ComplexF64[]
        for (expanded, scale) in expansions
            scaled_couplings = [
                (scale * first(coupling), Base.tail(coupling)...)
                for coupling in couplings
            ]
            expanded_rows, expanded_columns, expanded_values =
                _discrete_operator_triplets(
                    basis,
                    expanded,
                    scaled_couplings,
                )
            append!(rows, expanded_rows)
            append!(columns, expanded_columns)
            append!(values, expanded_values)
        end
        return rows, columns, values
    end
    rows = Int[]
    columns = Int[]
    coefficient_type = isempty(couplings) ?
        Float64 :
        promote_type(
            Float64,
            (typeof(first(coupling)) for coupling in couplings)...,
        )
    values = coefficient_type[]
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.L
    ]
    diagonal_operator = all(
        operator -> operator in ('I', 'n', 'z', '|'),
        opstring,
    )
    diagonal = diagonal_operator ?
        zeros(coefficient_type, length(basis)) :
        coefficient_type[]
    for coupling in couplings
        coefficient = convert(coefficient_type, first(coupling))
        sites = coupling[2:end]
        actions = _operator_actions(basis, opstring, sites)
        for column in eachindex(basis.encoded_states)
            encoded = basis.encoded_states[column]
            amplitude = coefficient
            alive = true
            for (op, site, species) in Iterators.reverse(actions)
                encoded, factor, alive = _apply_discrete_encoded_local(
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
            if diagonal_operator
                row == column || error("diagonal operator changed basis state")
                diagonal[column] += amplitude
            else
                push!(rows, row)
                push!(columns, column)
                push!(values, amplitude)
            end
        end
    end
    if diagonal_operator
        for index in eachindex(diagonal)
            iszero(diagonal[index]) && continue
            push!(rows, index)
            push!(columns, index)
            push!(values, diagonal[index])
        end
    end
    return rows, columns, values
end

function operator_matrix(
    basis::DiscreteBasis{K},
    opstring::AbstractString,
    couplings,
    ;
    sparse::Bool=false,
) where {K}
    if _has_symmetry(basis.symmetry)
        parent = _discrete_parent_basis(basis)
        rows, columns, values =
            _discrete_operator_triplets(parent, opstring, couplings)
        projected = _projected_triplet_matrix(
            basis.symmetry.projector,
            rows,
            columns,
            values,
        )
        return sparse ? projected : Matrix(projected)
    end
    rows, columns, values =
        _discrete_operator_triplets(basis, opstring, couplings)
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
    if _has_symmetry(basis.symmetry)
        parent = _discrete_parent_basis(basis)
        rows, columns, values =
            _discrete_operator_triplets(parent, opstring, couplings)
        return _accumulate_projected_triplets!(
            out,
            basis.symmetry.projector,
            rows,
            columns,
            values,
        )
    end
    rows, columns, values =
        _discrete_operator_triplets(basis, opstring, couplings)
    _accumulate_triplets!(out, rows, columns, values)
    return out
end

function _parent_basis_for_checks(basis::SpinBasis1D)
    _has_symmetry(basis.symmetry) || return basis
    return _spin_parent_basis(basis)
end

function _parent_basis_for_checks(basis::WideSpinBasis)
    _has_symmetry(basis.symmetry) || return basis
    return _wide_spin_parent_basis(basis)
end

function _parent_basis_for_checks(basis::DiscreteBasis{K}) where {K}
    _has_symmetry(basis.symmetry) || return basis
    return _discrete_parent_basis(basis)
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

function check_pcon(basis::WideSpinBasis, static, dynamic=Any[])
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
check_symm(basis::WideSpinBasis, static, dynamic=Any[]) =
    _check_projected_symmetry(basis, static, dynamic)
check_symm(basis::DiscreteBasis, static, dynamic=Any[]) =
    _check_projected_symmetry(basis, static, dynamic)

representative(basis::AbstractBasis, state::Integer) = state
normalization(basis::AbstractBasis, state::Integer) = one(Float64)
get_amp(basis::AbstractBasis, state::Integer) = one(Float64)

function _symmetry_column(data::SymmetryData, state::Integer)
    encoded = convert(eltype(data.parent_states), state)
    row = get(data.parent_lookup, encoded, 0)
    row == 0 && throw(ArgumentError("state is outside the parent particle sector"))
    columns, values = findnz(data.projector[row, :])
    isempty(columns) &&
        throw(ArgumentError("state has zero weight in this symmetry sector"))
    position = argmax(abs.(values))
    return columns[position], values[position]
end

function representative(
    basis::Union{SpinBasis1D,WideSpinBasis,DiscreteBasis},
    state::Integer,
)
    _has_symmetry(basis.symmetry) || return state
    column, _ = _symmetry_column(basis.symmetry, state)
    return basis.encoded_states[column]
end

function normalization(
    basis::Union{SpinBasis1D,WideSpinBasis,DiscreteBasis},
    state::Integer,
)
    _has_symmetry(basis.symmetry) || return one(Float64)
    _, amplitude = _symmetry_column(basis.symmetry, state)
    return inv(abs2(amplitude))
end

function get_amp(
    basis::Union{SpinBasis1D,WideSpinBasis,DiscreteBasis},
    state::Integer,
)
    _has_symmetry(basis.symmetry) || return one(Float64)
    _, amplitude = _symmetry_column(basis.symmetry, state)
    return amplitude
end
make_basis!(basis::AbstractBasis) = basis
function make_basis_blocks(
    basis::AbstractBasis;
    N_p::Union{Nothing,Integer}=nothing,
)
    length(basis) == 0 && return UnitRange{Int}[]
    applicable(states, basis) || return [1:length(basis)]
    encoded_states = states(basis)
    encoded_states isa AbstractVector || return [1:length(basis)]
    state_values = collect(encoded_states)
    site_count = hasproperty(basis, :L) ?
        Int(getproperty(basis, :L)) :
        hasproperty(basis, :N) ? Int(getproperty(basis, :N)) : 0
    site_count > 0 || return [1:length(basis)]
    prefix_bits = N_p === nothing ?
        clamp(floor(Int, log2(length(state_values) ÷ 2 + 1)), 0, site_count) :
        clamp(Int(N_p), 0, site_count)
    prefix_bits == 0 && return [1:length(basis)]
    shift = site_count - prefix_bits
    blocks = UnitRange{Int}[]
    first_index = firstindex(state_values)
    previous_prefix = state_values[first_index] >> shift
    for index in (first_index + 1):lastindex(state_values)
        prefix = state_values[index] >> shift
        prefix == previous_prefix && continue
        push!(blocks, first_index:(index - 1))
        first_index = index
        previous_prefix = prefix
    end
    push!(blocks, first_index:lastindex(state_values))
    return blocks
end
function project_to(
    basis::AbstractBasis,
    vector::AbstractVecOrMat;
    sparse::Bool=true,
    pcon::Bool=false,
)
    projector = pcon ?
        _pcon_projection_matrix(basis, eltype(vector)) :
        projection_matrix(basis, eltype(vector); sparse=true)
    size(vector, 1) == size(projector, 1) ||
        throw(DimensionMismatch(
            "the first vector dimension does not match the projection space",
        ))
    result = projector' * vector
    if sparse
        return SparseArrays.issparse(result) ?
            result :
            SparseArrays.sparse(result)
    end
    return SparseArrays.issparse(result) ? collect(result) : result
end
op_bra_ket(basis::DiscreteBasis, opstring, couplings) =
    operator_matrix(basis, opstring, couplings)

function _spin_full_operator(
    basis::SpinBasis1D,
    operator,
    ::Type{T},
) where {T<:Number}
    full_dimension = 1 << basis.L
    if operator isa AbstractMatrix
        size(operator) == (full_dimension, full_dimension) ||
            throw(DimensionMismatch("full operator has the wrong shape"))
        return Matrix{T}(operator)
    end
    is_entry = hasproperty(operator, :op) ||
        (
            (operator isa Tuple || operator isa AbstractVector) &&
            !isempty(operator) &&
            first(operator) isa AbstractString
        )
    entries = is_entry ? (operator,) : operator
    full_basis = SpinBasis1D(basis.L; pauli=basis.pauli)
    result = zeros(T, full_dimension, full_dimension)
    for entry in entries
        if hasproperty(entry, :op) && hasproperty(entry, :couplings)
            opstring = String(getproperty(entry, :op))
            couplings = getproperty(entry, :couplings)
        elseif (entry isa Tuple || entry isa AbstractVector) && length(entry) == 3
            opstring = String(entry[1])
            sites = Tuple(Int.(entry[2]))
            couplings = [(entry[3], sites...)]
        elseif (entry isa Tuple || entry isa AbstractVector) && length(entry) == 2
            opstring = String(entry[1])
            couplings = entry[2]
        else
            throw(ArgumentError(
                "operator entries must be (opstring, sites, coupling), " *
                "(opstring, couplings), or objects with op and couplings",
            ))
        end
        result .+= operator_matrix(full_basis, opstring, couplings)
    end
    return result
end

function op_shift_sector(
    target::SpinBasis1D,
    source::SpinBasis1D,
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
)
    target.L == source.L ||
        throw(ArgumentError("source and target bases must have the same length"))
    target.pauli == source.pauli ||
        throw(ArgumentError("source and target bases must use the same spin convention"))
    size(vector, 1) == length(source) ||
        throw(DimensionMismatch("the first vector dimension must equal source Ns"))
    coefficient_types = Type[eltype(vector)]
    if !(operator isa AbstractMatrix)
        entries = (
            (operator isa Tuple || operator isa AbstractVector) &&
            !isempty(operator) &&
            first(operator) isa AbstractString
        ) ? (operator,) : operator
        for entry in entries
            if hasproperty(entry, :couplings)
                append!(
                    coefficient_types,
                    typeof(first(coupling))
                    for coupling in getproperty(entry, :couplings)
                )
            elseif length(entry) == 3
                push!(coefficient_types, typeof(entry[3]))
            elseif length(entry) == 2
                append!(
                    coefficient_types,
                    typeof(first(coupling))
                    for coupling in entry[2]
                )
            end
        end
    else
        push!(coefficient_types, eltype(operator))
    end
    T = promote_type(ComplexF64, coefficient_types...)
    full_operator = _spin_full_operator(source, operator, T)
    source_projector = projection_matrix(source, T; sparse=true)
    target_projector = projection_matrix(target, T; sparse=true)
    result = target_projector' * (full_operator * (source_projector * vector))
    out === nothing && return result
    axes(out) == axes(result) ||
        throw(DimensionMismatch("out must have the same axes as the result"))
    copyto!(out, result)
    return out
end

function _normalized_shift_entries(operator)
    entries = (
        (operator isa Tuple || operator isa AbstractVector) &&
        !isempty(operator) &&
        first(operator) isa AbstractString
    ) ? (operator,) : operator
    normalized = Tuple{String,Tuple,Any}[]
    for entry in entries
        if hasproperty(entry, :op) && hasproperty(entry, :couplings)
            opstring = String(getproperty(entry, :op))
            for coupling in getproperty(entry, :couplings)
                push!(
                    normalized,
                    (opstring, Tuple(coupling[2:end]), first(coupling)),
                )
            end
        elseif (entry isa Tuple || entry isa AbstractVector) &&
               length(entry) == 3
            push!(
                normalized,
                (
                    String(entry[1]),
                    Tuple(Int.(entry[2])),
                    entry[3],
                ),
            )
        elseif (entry isa Tuple || entry isa AbstractVector) &&
               length(entry) == 2
            opstring = String(entry[1])
            for coupling in entry[2]
                push!(
                    normalized,
                    (opstring, Tuple(coupling[2:end]), first(coupling)),
                )
            end
        else
            throw(ArgumentError(
                "operator entries must contain an operator string, sites, and coupling",
            ))
        end
    end
    return normalized
end

function _parent_coordinates(symmetry::SymmetryData, vector)
    return symmetry.reduced ? symmetry.projector * vector : vector
end

function _reduced_coordinates(symmetry::SymmetryData, vector)
    return symmetry.reduced ? symmetry.projector' * vector : vector
end

function op_shift_sector(
    target::DiscreteBasis{K},
    source::DiscreteBasis{K},
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
) where {K}
    target.L == source.L && target.sps == source.sps ||
        throw(ArgumentError(
            "source and target bases must have the same local Hilbert space",
        ))
    size(vector, 1) == length(source) ||
        throw(DimensionMismatch(
            "the first vector dimension must equal source Ns",
        ))
    entries = _normalized_shift_entries(operator)
    coefficient_types = Type[eltype(vector)]
    append!(coefficient_types, typeof(entry[3]) for entry in entries)
    T = promote_type(ComplexF64, coefficient_types...)
    parent_input = _parent_coordinates(source.symmetry, vector)
    input_matrix = parent_input isa AbstractVector ?
        reshape(parent_input, :, 1) :
        parent_input
    parent_output = zeros(
        T,
        length(target.symmetry.parent_states),
        size(input_matrix, 2),
    )
    source_parent = _discrete_parent_basis(source)
    weights = UInt64[
        UInt64(source.sps)^(site - 1) for site in 1:source.L
    ]
    for (opstring, sites, coupling) in entries
        actions = _operator_actions(source_parent, opstring, sites)
        for (column, initial_state) in
            pairs(source.symmetry.parent_states)
            amplitude = complex(coupling)
            encoded = initial_state
            alive = true
            for (op, site, species) in Iterators.reverse(actions)
                encoded, factor, alive =
                    _apply_discrete_encoded_local(
                        source_parent,
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
            row = get(target.symmetry.parent_lookup, encoded, 0)
            row == 0 && continue
            @views parent_output[row, :] .+=
                amplitude .* input_matrix[column, :]
        end
    end
    result = _reduced_coordinates(target.symmetry, parent_output)
    vector isa AbstractVector && (result = vec(result))
    out === nothing && return result
    axes(out) == axes(result) ||
        throw(DimensionMismatch(
            "out must have the same axes as the result",
        ))
    copyto!(out, result)
    return out
end

function op_shift_sector(
    target::AbstractBasis,
    source::AbstractBasis,
    operator,
    vector::AbstractVecOrMat;
    out=nothing,
)
    throw(ArgumentError(
        "cross-sector operator application is not implemented for " *
        "$(typeof(target)) and $(typeof(source))",
    ))
end

function op_bra_ket(
    basis::SpinBasis1D,
    opstring::AbstractString,
    sites,
    coupling::Number,
    ::Type{T},
    ket_states;
    reduce_output::Bool=true,
) where {T<:Number}
    length(opstring) == length(sites) ||
        throw(ArgumentError("operator arity and sites differ"))
    kets = ket_states isa Integer ?
        UInt64[ket_states] :
        UInt64.(collect(ket_states))
    matrix_elements = zeros(T, length(kets))
    bras = similar(kets)
    for (position, ket) in pairs(kets)
        ket < (UInt64(1) << basis.L) ||
            throw(ArgumentError("ket state lies outside the full Hilbert space"))
        state = ket
        amplitude = complex(coupling)
        alive = true
        for index in length(opstring):-1:1
            state, factor, alive = _apply_spin_local(
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

function op_bra_ket(
    basis::DiscreteBasis,
    opstring::AbstractString,
    sites,
    coupling::Number,
    ::Type{T},
    ket_states;
    reduce_output::Bool=true,
) where {T<:Number}
    length(opstring) == length(sites) ||
        throw(ArgumentError("operator arity and sites differ"))
    actions = _operator_actions(basis, opstring, sites)
    weights = UInt64[
        UInt64(basis.sps)^(site - 1) for site in 1:basis.L
    ]
    kets = ket_states isa Integer ?
        UInt64[ket_states] :
        UInt64.(collect(ket_states))
    matrix_elements = zeros(T, length(kets))
    bras = similar(kets)
    for (position, ket) in pairs(kets)
        ket < UInt64(basis.sps)^basis.L ||
            throw(ArgumentError("ket state lies outside the full Hilbert space"))
        encoded = ket
        amplitude = complex(coupling)
        alive = true
        for (op, site, species) in Iterators.reverse(actions)
            encoded, factor, alive =
                _apply_discrete_encoded_local(
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
        bras[position] = encoded
        if alive
            matrix_elements[position] = convert(T, amplitude)
        end
    end
    reduce_output || return matrix_elements, bras, kets
    keep = .!iszero.(matrix_elements)
    return matrix_elements[keep], bras[keep], kets[keep]
end
