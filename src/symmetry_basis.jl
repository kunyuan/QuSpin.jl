"""
Internal description of a symmetry-reduced basis.

`projector` maps coordinates in the reduced basis to the particle-conserving
parent basis. Keeping this map explicit makes every reduced operator a true
`P' * O * P` projection and also gives `projection_matrix` exact semantics.
"""
struct SymmetryData
    parent_states::Vector{UInt64}
    parent_lookup::Dict{UInt64,Int}
    projector::SparseMatrixCSC{ComplexF64,Int}
    parent_occupations::Base.RefValue{Union{Nothing,Matrix{Int}}}
    blocks::Dict{Symbol,Any}
    reduced::Bool
end

function _identity_symmetry_data(
    states::Vector{UInt64},
    blocks,
    lookup::Dict{UInt64,Int}=Dict(
        state => index for (index, state) in pairs(states)
    ),
    ;
    parent_occupations=nothing,
)
    return SymmetryData(
        states,
        lookup,
        spzeros(ComplexF64, 0, 0),
        Ref{Union{Nothing,Matrix{Int}}}(parent_occupations),
        Dict{Symbol,Any}(blocks),
        false,
    )
end

const _MAX_DENSE_SYMMETRY_FALLBACK = 512

function _signed_permutation(
    states::Vector{UInt64},
    lookup::Dict{UInt64,Int},
    transform,
)
    rows = Vector{Int}(undef, length(states))
    values = Vector{ComplexF64}(undef, length(states))
    for (column, state) in pairs(states)
        transformed, phase = transform(state)
        row = get(lookup, transformed, 0)
        row == 0 && throw(ArgumentError(
            "symmetry transformation leaves the selected particle sector",
        ))
        rows[column] = row
        values[column] = phase
    end
    return sparse(rows, collect(1:length(states)), values, length(states), length(states))
end

function _cyclic_projector(
    states::Vector{UInt64},
    lookup::Dict{UInt64,Int},
    transform,
    eigenvalue::ComplexF64,
)
    visited = falses(length(states))
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    output_column = 0

    for start in eachindex(states)
        visited[start] && continue
        orbit = Int[]
        phases = ComplexF64[]
        current = start
        while true
            push!(orbit, current)
            visited[current] = true
            transformed, phase = transform(states[current])
            next_index = get(lookup, transformed, 0)
            next_index == 0 && throw(ArgumentError(
                "symmetry transformation leaves the selected particle sector",
            ))
            push!(phases, phase)
            next_index == start && break
            next_index in orbit && throw(ArgumentError(
                "symmetry map produced a non-cyclic orbit",
            ))
            current = next_index
        end

        coefficients = ones(ComplexF64, length(orbit))
        for index in 1:(length(orbit) - 1)
            coefficients[index + 1] =
                coefficients[index] * phases[index] / eigenvalue
        end
        closure = coefficients[end] * phases[end]
        isapprox(closure, eigenvalue * coefficients[1]; atol=2e-10, rtol=2e-10) ||
            continue

        output_column += 1
        normalization = norm(coefficients)
        for (row, coefficient) in zip(orbit, coefficients)
            push!(rows, row)
            push!(columns, output_column)
            push!(values, coefficient / normalization)
        end
    end
    return sparse(rows, columns, values, length(states), output_column)
end

function _intersect_eigenspace(
    projector::SparseMatrixCSC{ComplexF64,Int},
    symmetry::SparseMatrixCSC{ComplexF64,Int},
    eigenvalue::ComplexF64,
)
    size(projector, 2) == 0 && return projector
    represented_sparse = sparse(projector' * symmetry * projector)
    droptol!(represented_sparse, 5e-12)
    reduced_dimension = size(represented_sparse, 2)
    mapped_rows = zeros(Int, reduced_dimension)
    mapped_phases = zeros(ComplexF64, reduced_dimension)
    permutation_like = true
    for column in 1:reduced_dimension
        pointers = nzrange(represented_sparse, column)
        if length(pointers) != 1
            permutation_like = false
            break
        end
        pointer = first(pointers)
        mapped_rows[column] = rowvals(represented_sparse)[pointer]
        mapped_phases[column] = nonzeros(represented_sparse)[pointer]
        if !isapprox(abs(mapped_phases[column]), 1.0; atol=2e-10, rtol=2e-10)
            permutation_like = false
            break
        end
    end
    permutation_like &=
        length(unique(mapped_rows)) == reduced_dimension

    if permutation_like
        visited = falses(reduced_dimension)
        rows = Int[]
        columns = Int[]
        values = ComplexF64[]
        output_column = 0
        for start in 1:reduced_dimension
            visited[start] && continue
            orbit = Int[]
            phases = ComplexF64[]
            current = start
            while true
                push!(orbit, current)
                visited[current] = true
                push!(phases, mapped_phases[current])
                next_index = mapped_rows[current]
                next_index == start && break
                if visited[next_index]
                    permutation_like = false
                    break
                end
                current = next_index
            end
            permutation_like || break
            coefficients = ones(ComplexF64, length(orbit))
            for index in 1:(length(orbit) - 1)
                coefficients[index + 1] =
                    coefficients[index] * phases[index] / eigenvalue
            end
            closure = coefficients[end] * phases[end]
            isapprox(
                closure,
                eigenvalue * coefficients[1];
                atol=2e-10,
                rtol=2e-10,
            ) || continue
            output_column += 1
            normalization = norm(coefficients)
            for (row, coefficient) in zip(orbit, coefficients)
                push!(rows, row)
                push!(columns, output_column)
                push!(values, coefficient / normalization)
            end
        end
        if permutation_like
            output_column == 0 &&
                return spzeros(ComplexF64, size(projector, 1), 0)
            reduced_projector = sparse(
                rows,
                columns,
                values,
                reduced_dimension,
                output_column,
            )
            result = sparse(projector * reduced_projector)
            droptol!(result, 5e-14)
            return result
        end
    end

    reduced_dimension <= _MAX_DENSE_SYMMETRY_FALLBACK ||
        throw(ArgumentError(
            "the requested symmetry intersection requires a dense nullspace " *
            "of dimension $reduced_dimension; the guarded limit is " *
            "$_MAX_DENSE_SYMMETRY_FALLBACK. Supply permutation-like symmetry " *
            "maps or split the symmetry sector before intersecting it.",
        ))
    represented = Matrix(represented_sparse)
    residual = represented - eigenvalue * I
    vectors = nullspace(residual; atol=2e-10, rtol=2e-10)
    isempty(vectors) && return spzeros(ComplexF64, size(projector, 1), 0)
    result = sparse(projector * vectors)
    droptol!(result, 5e-14)
    return result
end

function _representative_states(
    parent_states::Vector{UInt64},
    projector::SparseMatrixCSC,
)
    representatives = UInt64[]
    for column in axes(projector, 2)
        rows, values = findnz(@view projector[:, column])
        isempty(rows) && continue
        largest = firstindex(values)
        for index in (firstindex(values) + 1):lastindex(values)
            abs(values[index]) > abs(values[largest]) &&
                (largest = index)
        end
        index = rows[largest]
        push!(representatives, parent_states[index])
    end
    return representatives
end

function _finalize_symmetry_data(
    parent_states::Vector{UInt64},
    projector::SparseMatrixCSC,
    blocks,
    ;
    parent_occupations=nothing,
)
    size(projector, 2) > 0 ||
        throw(ArgumentError("the requested symmetry sector is empty"))
    gram = sparse(projector' * projector)
    residual = gram - spdiagm(0 => ones(ComplexF64, size(gram, 1)))
    maximum(abs, nonzeros(residual); init=0.0) <= 4e-10 ||
        throw(ArgumentError("internal symmetry projector is not orthonormal"))
    return SymmetryData(
        copy(parent_states),
        Dict(state => index for (index, state) in pairs(parent_states)),
        SparseMatrixCSC{ComplexF64,Int}(projector),
        Ref{Union{Nothing,Matrix{Int}}}(parent_occupations),
        Dict{Symbol,Any}(blocks),
        true,
    )
end

function _permutation_phase(occupied_modes::Vector{Int}, permutation::Vector{Int})
    mapped = [permutation[mode] for mode in occupied_modes]
    inversions = 0
    for left in eachindex(mapped), right in (left + 1):length(mapped)
        left < right || continue
        inversions += mapped[left] > mapped[right]
    end
    return iseven(inversions) ? 1.0 + 0im : -1.0 + 0im
end

function _site_permutation_transform(
    state::UInt64,
    occupations::Vector{Int},
    sps::Int,
    permutation::Vector{Int};
    fermionic::Bool=false,
    spinful::Bool=false,
)
    transformed = zeros(Int, length(occupations))
    for site in eachindex(occupations)
        transformed[permutation[site]] = occupations[site]
    end
    phase = 1.0 + 0im
    if fermionic
        if spinful
            occupied = Int[]
            mode_permutation = vcat(permutation, length(permutation) .+ permutation)
            for site in eachindex(occupations)
                occupations[site] & 1 == 1 && push!(occupied, site)
                occupations[site] & 2 == 2 &&
                    push!(occupied, length(occupations) + site)
            end
            sort!(occupied)
            phase = _permutation_phase(occupied, mode_permutation)
        else
            occupied = findall(!iszero, occupations)
            phase = _permutation_phase(occupied, permutation)
        end
    end
    encoded = UInt64(sum(
        transformed[site] * sps^(site - 1)
        for site in eachindex(transformed)
    ))
    return encoded, phase
end

function _fermion_site_permutation_phase(
    state::UInt64,
    sps::Int,
    weights::Vector{UInt64},
    permutation::Vector{Int};
    spinful::Bool=false,
)
    inversions = 0
    modes = spinful ? 2 * length(permutation) : length(permutation)
    for left_mode in 1:(modes - 1)
        left_site = mod1(left_mode, length(permutation))
        left_digit = Int((state ÷ weights[left_site]) % UInt64(sps))
        left_occupied = spinful ?
            (left_mode <= length(permutation) ?
             (left_digit & 1 == 1) : (left_digit & 2 == 2)) :
            !iszero(left_digit)
        left_occupied || continue
        mapped_left = left_mode <= length(permutation) ?
            permutation[left_site] :
            length(permutation) + permutation[left_site]
        for right_mode in (left_mode + 1):modes
            right_site = mod1(right_mode, length(permutation))
            right_digit =
                Int((state ÷ weights[right_site]) % UInt64(sps))
            right_occupied = spinful ?
                (right_mode <= length(permutation) ?
                 (right_digit & 1 == 1) : (right_digit & 2 == 2)) :
                !iszero(right_digit)
            right_occupied || continue
            mapped_right = right_mode <= length(permutation) ?
                permutation[right_site] :
                length(permutation) + permutation[right_site]
            inversions += mapped_left > mapped_right
        end
    end
    return iseven(inversions) ? 1.0 + 0im : -1.0 + 0im
end

function _site_permutation_transform(
    state::UInt64,
    sps::Int,
    permutation::Vector{Int},
    weights::Vector{UInt64};
    fermionic::Bool=false,
    spinful::Bool=false,
)
    encoded = zero(UInt64)
    for site in eachindex(permutation)
        occupation = (state ÷ weights[site]) % UInt64(sps)
        encoded += occupation * weights[permutation[site]]
    end
    phase = fermionic ?
        _fermion_site_permutation_phase(
            state,
            sps,
            weights,
            permutation;
            spinful,
        ) :
        1.0 + 0im
    return encoded, phase
end

function _full_projection_matrix(
    states::Vector{UInt64},
    symmetry::SymmetryData,
    full_dimension::Int,
    ::Type{T},
    ;
    sparse_output::Bool=false,
) where {T<:Number}
    if !symmetry.reduced
        rows = Int[Int(state) + 1 for state in symmetry.parent_states]
        values = ones(T, length(rows))
        projected = sparse(
            rows,
            collect(1:length(rows)),
            values,
            full_dimension,
            length(rows),
        )
        return sparse_output ? projected : Matrix(projected)
    end
    parent_projector = symmetry.projector
    rows = Int[
        Int(symmetry.parent_states[row]) + 1
        for row in rowvals(parent_projector)
    ]
    values = if T <: Real
        maximum(abs ∘ imag, nonzeros(parent_projector); init=0.0) <= 2e-12 ||
            throw(ArgumentError(
                "this momentum sector requires a complex projection dtype",
            ))
        T[real(value) for value in nonzeros(parent_projector)]
    else
        T[value for value in nonzeros(parent_projector)]
    end
    projected = SparseMatrixCSC(
        full_dimension,
        size(parent_projector, 2),
        copy(parent_projector.colptr),
        rows,
        values,
    )
    return sparse_output ? projected : Matrix(projected)
end

function _project_from_full(
    symmetry::SymmetryData,
    vector::AbstractVecOrMat,
    full_dimension::Int,
)
    if SparseArrays.issparse(vector)
        projected = _full_projection_matrix(
            symmetry.parent_states,
            symmetry,
            full_dimension,
            eltype(vector);
            sparse_output=true,
        )
        return projected * vector
    end
    requires_complex = symmetry.reduced &&
        maximum(abs ∘ imag, nonzeros(symmetry.projector); init=0.0) > 2e-12
    if eltype(vector) <: Real && requires_complex
        throw(ArgumentError(
            "this momentum sector requires a complex projection dtype",
        ))
    end
    parent_coordinates = if symmetry.reduced
        projected = symmetry.projector * vector
        eltype(vector) <: Real ? real.(projected) : projected
    else
        vector
    end
    rows = Int[Int(state) + 1 for state in symmetry.parent_states]
    if vector isa AbstractVector
        result = zeros(eltype(parent_coordinates), full_dimension)
        result[rows] = parent_coordinates
        return result
    end
    result = zeros(
        eltype(parent_coordinates),
        full_dimension,
        size(vector, 2),
    )
    result[rows, :] = parent_coordinates
    return result
end

function _accumulate_projected_triplets!(
    output,
    projector::SparseMatrixCSC,
    parent_rows,
    parent_columns,
    parent_values,
)
    adjoint_projector = sparse(adjoint(projector))
    reduced_rows = rowvals(adjoint_projector)
    coefficients = nonzeros(adjoint_projector)
    @inbounds for entry in eachindex(parent_values)
        parent_row = parent_rows[entry]
        parent_column = parent_columns[entry]
        matrix_value = parent_values[entry]
        for left_pointer in nzrange(adjoint_projector, parent_row)
            reduced_row = reduced_rows[left_pointer]
            left_value = coefficients[left_pointer]
            for right_pointer in nzrange(adjoint_projector, parent_column)
                reduced_column = reduced_rows[right_pointer]
                right_value = conj(coefficients[right_pointer])
                output[reduced_row, reduced_column] +=
                    left_value * matrix_value * right_value
            end
        end
    end
    return output
end

_has_symmetry(data::SymmetryData) = data.reduced

function _basis_requires_complex(basis)
    hasfield(typeof(basis), :symmetry) || return false
    symmetry = getfield(basis, :symmetry)
    symmetry.reduced || return false
    return any(value -> abs(imag(value)) > 2e-12, nonzeros(symmetry.projector))
end
