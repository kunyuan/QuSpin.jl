struct TensorBasis{L<:AbstractBasis,R<:AbstractBasis} <: AbstractBasis
    basis_left::L
    basis_right::R
end

TensorBasis() =
    throw(ArgumentError("TensorBasis requires at least two factors"))
TensorBasis(::AbstractBasis) =
    throw(ArgumentError("TensorBasis requires at least two factors"))
TensorBasis(
    first::AbstractBasis,
    second::AbstractBasis,
    third::AbstractBasis,
    rest::AbstractBasis...,
) = TensorBasis(TensorBasis(first, second), third, rest...)

Base.length(basis::TensorBasis) =
    length(basis.basis_left) * length(basis.basis_right)

_tensor_factors(basis::AbstractBasis) = AbstractBasis[basis]
_tensor_factors(basis::TensorBasis) =
    vcat(_tensor_factors(basis.basis_left), _tensor_factors(basis.basis_right))

function Base.getproperty(basis::TensorBasis, name::Symbol)
    factors = name in (:N, :sps, :blocks, :operators) ?
        _tensor_factors(basis) :
        nothing
    name === :N && return Tuple(getproperty(factor, :N) for factor in factors)
    name === :Ns && return length(basis)
    name === :sps &&
        return Tuple(getproperty(factor, :sps) for factor in factors)
    name === :blocks && return length(factors) == 2 ?
        Dict(
            :left => getproperty(factors[1], :blocks),
            :right => getproperty(factors[2], :blocks),
        ) :
        Dict(
            index => getproperty(factor, :blocks)
            for (index, factor) in pairs(factors)
        )
    name === :operators &&
        return Tuple(getproperty(factor, :operators) for factor in factors)
    return getfield(basis, name)
end

projection_matrix(
    basis::TensorBasis,
    ::Type{T}=Float64;
    sparse::Bool=false,
    pcon::Bool=false,
) where {T<:Number} =
    kron(
        projection_matrix(basis.basis_left, T; sparse, pcon),
        projection_matrix(basis.basis_right, T; sparse, pcon),
    )

_full_projection_dimension(basis::TensorBasis) =
    _full_projection_dimension(basis.basis_left) *
    _full_projection_dimension(basis.basis_right)
_projection_output_dimension(basis::TensorBasis, pcon::Bool) =
    _projection_output_dimension(basis.basis_left, pcon) *
    _projection_output_dimension(basis.basis_right, pcon)
_pcon_projection_matrix(
    basis::TensorBasis,
    ::Type{T},
) where {T<:Number} =
    kron(
        _pcon_projection_matrix(basis.basis_left, T),
        _pcon_projection_matrix(basis.basis_right, T),
    )

function project_from(
    basis::TensorBasis,
    vector::AbstractVecOrMat;
    kwargs...,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    sparse_projection = get(kwargs, :sparse, true)
    pcon_projection = get(kwargs, :pcon, false)
    left_dimension = length(basis.basis_left)
    right_dimension = length(basis.basis_right)
    function project_column(column)
        coefficients = reshape(column, right_dimension, left_dimension)
        right_expanded = project_from(
            basis.basis_right,
            coefficients;
            sparse=sparse_projection,
            pcon=pcon_projection,
        )
        left_expanded = project_from(
            basis.basis_left,
            transpose(right_expanded);
            sparse=sparse_projection,
            pcon=pcon_projection,
        )
        column_result = vec(transpose(left_expanded))
        return sparse_projection ?
            SparseArrays.sparse(column_result) :
            collect(column_result)
    end
    if vector isa AbstractVector
        return project_column(vector)
    end
    output_dimension =
        _projection_output_dimension(basis, pcon_projection)
    isempty(axes(vector, 2)) &&
        return sparse_projection ?
            spzeros(eltype(vector), output_dimension, 0) :
            Matrix{eltype(vector)}(undef, output_dimension, 0)
    first_column = first(axes(vector, 2))
    first_projected =
        project_column(@view(vector[:, first_column]))
    output = Matrix{eltype(first_projected)}(
        undef,
        output_dimension,
        size(vector, 2),
    )
    output[:, first_column] = first_projected
    for column in Iterators.drop(axes(vector, 2), 1)
        output[:, column] = project_column(@view(vector[:, column]))
    end
    return sparse_projection ? SparseArrays.sparse(output) : output
end

get_vec(basis::TensorBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)

function _expanded_operator_strings(
    basis::TensorBasis,
    opstring::AbstractString,
)
    factors = _tensor_factors(basis)
    pieces = split(opstring, "|"; keepempty=true)
    length(pieces) == length(factors) ||
        throw(ArgumentError(
            "tensor operator strings require one piece per factor",
        ))
    expansions = Tuple{String,ComplexF64}[("", 1.0 + 0im)]
    for (index, (factor, piece)) in
        enumerate(zip(factors, pieces))
        local_expansions = _expanded_operator_strings(factor, piece)
        expansions = [
            (
                prefix *
                (index == 1 ? "" : "|") *
                local_operator,
                coefficient * local_scale,
            )
            for (prefix, coefficient) in expansions
            for (local_operator, local_scale) in local_expansions
        ]
    end
    return expansions
end

expanded_form(basis::TensorBasis, static=Any[], dynamic=Any[]) = (
    _expanded_form_entries(basis, static, false),
    _expanded_form_entries(basis, dynamic, true),
)

function state_index(
    basis::TensorBasis,
    left_index::Integer,
    right_index::Integer,
)
    checkbounds(1:length(basis.basis_left), left_index)
    checkbounds(1:length(basis.basis_right), right_index)
    return (left_index - 1) * length(basis.basis_right) + right_index
end

function state_index(basis::TensorBasis, indices::Integer...)
    factors = _tensor_factors(basis)
    length(indices) == length(factors) ||
        throw(ArgumentError("one state index is required per tensor factor"))
    dimensions = length.(factors)
    linear = 1
    stride = 1
    for factor in length(factors):-1:1
        checkbounds(1:dimensions[factor], indices[factor])
        linear += (indices[factor] - 1) * stride
        stride *= dimensions[factor]
    end
    return linear
end

function _tensor_multi_index(index::Int, dimensions)
    remaining = index - 1
    result = Vector{Int}(undef, length(dimensions))
    for factor in length(dimensions):-1:1
        result[factor] = mod(remaining, dimensions[factor]) + 1
        remaining = div(remaining, dimensions[factor])
    end
    return result
end

function _selected_tensor_index(indices, dimensions, selected)
    linear = 1
    stride = 1
    for factor in Iterators.reverse(selected)
        linear += (indices[factor] - 1) * stride
        stride *= dimensions[factor]
    end
    return linear
end

function _tensor_selective_reductions(
    basis::TensorBasis,
    state::AbstractVecOrMat,
    selected_A,
    return_rdm,
)
    factors = _tensor_factors(basis)
    dimensions = length.(factors)
    selected = sort!(unique(Int.(collect(selected_A))))
    all(factor -> 1 <= factor <= length(factors), selected) ||
        throw(ArgumentError("tensor subsystem indices lie outside the factor list"))
    selected_B = setdiff(collect(eachindex(factors)), selected)
    dimension_A = prod(dimensions[selected]; init=1)
    dimension_B = prod(dimensions[selected_B]; init=1)
    indices = [
        _tensor_multi_index(linear, dimensions)
        for linear in 1:length(basis)
    ]
    indices_A = [
        _selected_tensor_index(value, dimensions, selected)
        for value in indices
    ]
    indices_B = [
        _selected_tensor_index(value, dimensions, selected_B)
        for value in indices
    ]
    if state isa AbstractVector
        length(state) == length(basis) ||
            throw(DimensionMismatch("state length must equal Ns"))
        coefficients =
            zeros(eltype(state), dimension_A, dimension_B)
        for linear in eachindex(state)
            coefficients[indices_A[linear], indices_B[linear]] = state[linear]
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
    return return_rdm in (:A, "A", :left, "left") ? rho_A :
           return_rdm in (:B, "B", :right, "right") ? rho_B :
           return_rdm in (:both, "both") ? (rho_A, rho_B) :
           throw(ArgumentError("return_rdm must be A, B, left, right, or both"))
end

function _tensor_reductions(
    basis::TensorBasis,
    state::AbstractVector,
    return_rdm=:both,
)
    length(state) == length(basis) ||
        throw(DimensionMismatch("state length must equal Ns"))
    coefficients = reshape(
        state,
        length(basis.basis_right),
        length(basis.basis_left),
    )
    if return_rdm in (:A, "A", :left, "left")
        return coefficients' * coefficients
    elseif return_rdm in (:B, "B", :right, "right")
        return coefficients * coefficients'
    elseif return_rdm in (:both, "both")
        return coefficients' * coefficients, coefficients * coefficients'
    end
    throw(ArgumentError("return_rdm must be A, B, left, right, or both"))
end

function _tensor_reductions(
    basis::TensorBasis,
    state::AbstractMatrix,
    return_rdm=:both,
)
    size(state) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("density matrix must match Ns"))
    left_dimension = length(basis.basis_left)
    right_dimension = length(basis.basis_right)
    need_left = return_rdm in (:A, "A", :left, "left", :both, "both")
    need_right = return_rdm in (:B, "B", :right, "right", :both, "both")
    need_left || need_right ||
        throw(ArgumentError("return_rdm must be A, B, left, right, or both"))
    rho_left = need_left ?
        zeros(eltype(state), left_dimension, left_dimension) :
        nothing
    rho_right = need_right ?
        zeros(eltype(state), right_dimension, right_dimension) :
        nothing
    reshaped = reshape(
        state,
        right_dimension,
        left_dimension,
        right_dimension,
        left_dimension,
    )
    if need_left
        for left_row in 1:left_dimension, left_column in 1:left_dimension,
            right in 1:right_dimension
            rho_left[left_row, left_column] +=
                reshaped[right, left_row, right, left_column]
        end
    end
    if need_right
        for right_row in 1:right_dimension, right_column in 1:right_dimension,
            left in 1:left_dimension
            rho_right[right_row, right_column] +=
                reshaped[right_row, left, right_column, left]
        end
    end
    return need_left && need_right ? (rho_left, rho_right) :
           need_left ? rho_left : rho_right
end

function partial_trace(
    basis::TensorBasis,
    state::AbstractVecOrMat;
    sub_sys_A=nothing,
    return_rdm=:A,
    enforce_pure::Bool=false,
    kwargs...,
)
    if state isa AbstractMatrix &&
       enforce_pure &&
       size(state, 1) == length(basis)
        reductions = [
            partial_trace(
                basis,
                @view(state[:, index]);
                sub_sys_A,
                return_rdm,
            )
            for index in axes(state, 2)
        ]
        if return_rdm in (:both, "both")
            return (
                cat((pair[1] for pair in reductions)...; dims=3),
                cat((pair[2] for pair in reductions)...; dims=3),
            )
        end
        return cat(reductions...; dims=3)
    end
    if sub_sys_A !== nothing || length(_tensor_factors(basis)) > 2
        selected = sub_sys_A === nothing ? [1] : sub_sys_A
        return _tensor_selective_reductions(
            basis,
            state,
            selected,
            return_rdm,
        )
    end
    return _tensor_reductions(basis, state, return_rdm)
end

function ent_entropy(
    basis::TensorBasis,
    state::AbstractVecOrMat;
    sub_sys_A=nothing,
    density::Bool=false,
    return_rdm=nothing,
    enforce_pure::Bool=false,
    return_rdm_EVs::Bool=false,
    alpha::Real=1.0,
    kwargs...,
)
    if state isa AbstractVector
        rho_left, rho_right = partial_trace(
            basis,
            state;
            sub_sys_A,
            return_rdm=:both,
        )
        probabilities = _density_eigenvalues(rho_left)
        entropy = _entropy_from_probabilities(probabilities, alpha)
        result = Dict{String,Any}("Sent_A" => entropy)
        if return_rdm in (:A, "A", :both, "both")
            result["rdm_A"] = rho_left
        end
        if return_rdm in (:B, "B", :both, "both")
            result["Sent_B"] = entropy
            result["rdm_B"] = rho_right
        end
        return_rdm_EVs && (result["p_A"] = probabilities)
        return result
    end
    need_right = return_rdm in (:B, "B", :both, "both")
    rho_left, rho_right = if need_right
        partial_trace(
            basis,
            state;
            sub_sys_A,
            return_rdm=:both,
            enforce_pure,
        )
    else
        partial_trace(
            basis,
            state;
            sub_sys_A,
            return_rdm=:A,
            enforce_pure,
        ), nothing
    end
    if ndims(rho_left) == 3
        probabilities_A = [
            _density_eigenvalues(@view(rho_left[:, :, index]))
            for index in axes(rho_left, 3)
        ]
        result = Dict{String,Any}(
            "Sent_A" => [
                _entropy_from_probabilities(probabilities, alpha)
                for probabilities in probabilities_A
            ],
        )
        if return_rdm in (:A, "A", :both, "both")
            result["rdm_A"] = rho_left
        end
        if need_right
            result["Sent_B"] = [
                _entropy_from_density(
                    @view(rho_right[:, :, index]),
                    alpha,
                )
                for index in axes(rho_right, 3)
            ]
            result["rdm_B"] = rho_right
        end
        return_rdm_EVs &&
            (result["p_A"] = reduce(hcat, probabilities_A))
        return result
    end
    probabilities_A = _density_eigenvalues(rho_left)
    result = Dict{String,Any}(
        "Sent_A" => _entropy_from_probabilities(probabilities_A, alpha),
    )
    if return_rdm in (:A, "A", :both, "both")
        result["rdm_A"] = rho_left
    end
    if need_right
        result["Sent_B"] = _entropy_from_density(rho_right, alpha)
        result["rdm_B"] = rho_right
    end
    return_rdm_EVs && (result["p_A"] = probabilities_A)
    return result
end

function _tensor_operator_factors(
    basis::TensorBasis,
    left_op,
    right_op,
    coupling,
    sparse_output::Bool,
)
    sites = coupling[2:end]
    length(sites) == length(left_op) + length(right_op) ||
        throw(ArgumentError("operator arity and sites differ"))
    left_sites = sites[1:length(left_op)]
    right_sites = sites[(length(left_op) + 1):end]
    left = isempty(left_op) ?
        sparse_output ?
        spdiagm(
            0 => fill(
                complex(first(coupling)),
                length(basis.basis_left),
            ),
        ) :
        Matrix{ComplexF64}(
            I,
            length(basis.basis_left),
            length(basis.basis_left),
        ) * first(coupling) :
        operator_matrix(
            basis.basis_left,
            left_op,
            [(first(coupling), left_sites...)],
            sparse=sparse_output,
        )
    right = isempty(right_op) ?
        sparse_output ?
        spdiagm(0 => ones(ComplexF64, length(basis.basis_right))) :
        Matrix{ComplexF64}(
            I,
            length(basis.basis_right),
            length(basis.basis_right),
        ) :
        operator_matrix(
            basis.basis_right,
            right_op,
            [(one(first(coupling)), right_sites...)],
            sparse=sparse_output,
        )
    return left, right
end

function operator_matrix(
    basis::TensorBasis,
    opstring::AbstractString,
    couplings,
    ;
    sparse::Bool=false,
)
    pieces = split(opstring, "|"; keepempty=true)
    factors = _tensor_factors(basis)
    length(pieces) == length(factors) ||
        throw(ArgumentError("tensor operator strings require one piece per factor"))
    matrix = sparse ?
        spzeros(ComplexF64, length(basis), length(basis)) :
        zeros(ComplexF64, length(basis), length(basis))
    for coupling in couplings
        sites = Base.tail(coupling)
        sum(length, pieces) == length(sites) ||
            throw(ArgumentError("operator arity and sites differ"))
        cursor = 1
        local_matrices = Any[]
        for (factor_index, (factor, piece)) in enumerate(zip(factors, pieces))
            local_sites = sites[cursor:(cursor + length(piece) - 1)]
            cursor += length(piece)
            coefficient = factor_index == 1 ? first(coupling) : one(first(coupling))
            local_matrix = if isempty(piece)
                if sparse
                    spdiagm(0 => fill(complex(coefficient), length(factor)))
                else
                    Matrix{ComplexF64}(I, length(factor), length(factor)) .* coefficient
                end
            else
                operator_matrix(
                    factor,
                    piece,
                    [(coefficient, local_sites...)];
                    sparse,
                )
            end
            push!(local_matrices, local_matrix)
        end
        product_matrix = reduce(kron, local_matrices)
        sparse ? (matrix += product_matrix) : (matrix .+= product_matrix)
    end
    return matrix
end

function _accumulate_kron!(out, left::SparseMatrixCSC, right::SparseMatrixCSC)
    right_dimension = size(right, 1)
    left_rows = rowvals(left)
    left_values = nonzeros(left)
    right_rows = rowvals(right)
    right_values = nonzeros(right)
    for left_column in axes(left, 2)
        for left_pointer in nzrange(left, left_column)
            output_row_offset =
                (left_rows[left_pointer] - 1) * right_dimension
            output_column_offset =
                (left_column - 1) * right_dimension
            for right_column in axes(right, 2)
                for right_pointer in nzrange(right, right_column)
                    out[
                        output_row_offset + right_rows[right_pointer],
                        output_column_offset + right_column,
                    ] +=
                        left_values[left_pointer] *
                        right_values[right_pointer]
                end
            end
        end
    end
    return out
end

function inplace_op!(out, basis::TensorBasis, opstring, couplings)
    size(out) == (length(basis), length(basis)) ||
        throw(DimensionMismatch("out must have shape (Ns,Ns)"))
    out .+= operator_matrix(basis, opstring, couplings; sparse=true)
    return out
end

struct PhotonBasis{
    P<:AbstractBasis,
    B<:TensorBasis,
    S<:SparseMatrixCSC{ComplexF64,Int},
} <: AbstractBasis
    basis_particle::P
    basis_photon::BosonBasis1D
    tensor::B
    Nph::Int
    Ntot::Union{Nothing,Int}
    selection::S
    photon_numbers::Vector{Int}
end

_particle_excitation_count(basis::SpinBasis1D, state::Integer) =
    count_ones(UInt64(state))
function _particle_excitation_count(basis::DiscreteBasis{K}, state::Integer) where {K}
    occupations = _digits(state, basis.L, basis.sps)
    return K === :spinful_fermion ?
        sum(count_ones(UInt(digit)) for digit in occupations) :
        sum(occupations)
end
function _particle_excitation_count(basis::AbstractBasis, state::Integer)
    hasproperty(basis, :base) &&
        return _particle_excitation_count(getproperty(basis, :base), state)
    throw(ArgumentError(
        "particle excitation counting is not defined for $(typeof(basis))",
    ))
end

function PhotonBasis(
    basis_constructor,
    constructor_args...;
    Nph=nothing,
    Ntot=nothing,
    kwargs...,
)
    Nph === nothing && Ntot === nothing &&
        throw(ArgumentError("PhotonBasis requires Nph or Ntot"))
    Nph !== nothing && Ntot !== nothing &&
        throw(ArgumentError("specify only one of Nph and Ntot"))
    cutoff = Nph === nothing ? Int(Ntot) : Int(Nph)
    cutoff >= 0 || throw(ArgumentError("photon cutoff must be nonnegative"))
    particle = basis_constructor(constructor_args...; kwargs...)
    photon = BosonBasis1D(1; sps=cutoff + 1)
    tensor = TensorBasis(particle, photon)
    if Ntot === nothing
        dimension = length(tensor)
        selection = spdiagm(0 => ones(ComplexF64, dimension))
        photon_numbers = repeat(collect(0:cutoff), length(particle))
        return PhotonBasis(
            particle,
            photon,
            tensor,
            cutoff,
            nothing,
            selection,
            photon_numbers,
        )
    end
    particle_states = states(particle)
    rows = Int[]
    photon_numbers = Int[]
    for (particle_index, state) in pairs(particle_states)
        excitations = _particle_excitation_count(particle, state)
        excitations <= cutoff || continue
        photon_number = cutoff - excitations
        push!(
            rows,
            state_index(tensor, particle_index, photon_number + 1),
        )
        push!(photon_numbers, photon_number)
    end
    columns = collect(1:length(rows))
    selection = sparse(
        rows,
        columns,
        ones(ComplexF64, length(rows)),
        length(tensor),
        length(rows),
    )
    return PhotonBasis(
        particle,
        photon,
        tensor,
        cutoff,
        cutoff,
        selection,
        photon_numbers,
    )
end

Base.length(basis::PhotonBasis) = size(basis.selection, 2)
_full_projection_dimension(basis::PhotonBasis) =
    _full_projection_dimension(basis.tensor)
_projection_output_dimension(basis::PhotonBasis, pcon::Bool) =
    _projection_output_dimension(basis.tensor, pcon)
_pcon_projection_matrix(
    basis::PhotonBasis,
    ::Type{T},
) where {T<:Number} =
    _pcon_projection_matrix(basis.tensor, T) *
    SparseMatrixCSC{T,Int}(basis.selection)

function Base.getproperty(basis::PhotonBasis, name::Symbol)
    name in (:basis_left, :basis_particle) && return getfield(basis, :basis_particle)
    name in (:basis_right, :basis_photon) && return getfield(basis, :basis_photon)
    name === :N && return getproperty(getfield(basis, :basis_particle), :N) + 1
    name === :Ns && return length(basis)
    name === :chain_N && return getproperty(getfield(basis, :basis_particle), :N)
    name === :chain_Ns && return length(getfield(basis, :basis_particle))
    name === :particle_N && return getproperty(getfield(basis, :basis_particle), :N)
    name === :particle_Ns && return length(getfield(basis, :basis_particle))
    name === :particle_sps && return getproperty(getfield(basis, :basis_particle), :sps)
    name === :photon_Ns && return length(getfield(basis, :basis_photon))
    name === :photon_sps && return getproperty(getfield(basis, :basis_photon), :sps)
    name === :sps && return (
        getproperty(getfield(basis, :basis_particle), :sps),
        getproperty(getfield(basis, :basis_photon), :sps),
    )
    name === :blocks && return getfield(basis, :Ntot) === nothing ?
        Dict(:Nph => getfield(basis, :Nph)) :
        Dict(:Ntot => getfield(basis, :Ntot))
    name === :operators && return getproperty(getfield(basis, :tensor), :operators)
    return getfield(basis, name)
end

projection_matrix(
    basis::PhotonBasis,
    ::Type{T}=Float64;
    sparse::Bool=false,
    pcon::Bool=false,
) where {T<:Number} =
    begin
        projected =
            projection_matrix(basis.tensor, T; sparse=true, pcon) *
            SparseMatrixCSC{T,Int}(basis.selection)
        sparse ? projected : Matrix(projected)
    end
function project_from(
    basis::PhotonBasis,
    vector::AbstractVecOrMat;
    kwargs...,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    return project_from(basis.tensor, basis.selection * vector; kwargs...)
end
get_vec(basis::PhotonBasis, vector::AbstractVecOrMat; kwargs...) =
    get_vec(basis.tensor, vector; kwargs...)
expanded_form(basis::PhotonBasis, static=Any[], dynamic=Any[]) =
    expanded_form(basis.tensor, static, dynamic)
partial_trace(basis::PhotonBasis, state::AbstractVecOrMat; kwargs...) =
    partial_trace(
        basis.tensor,
        state isa AbstractVector || (
            state isa AbstractMatrix &&
            size(state, 1) == length(basis) &&
            size(state, 2) != length(basis)
        ) ?
            basis.selection * state :
            basis.selection * state * basis.selection';
        kwargs...,
    )
ent_entropy(basis::PhotonBasis, state::AbstractVecOrMat; kwargs...) =
    ent_entropy(
        basis.tensor,
        state isa AbstractVector || (
            state isa AbstractMatrix &&
            size(state, 1) == length(basis) &&
            size(state, 2) != length(basis)
        ) ?
            basis.selection * state :
            basis.selection * state * basis.selection';
        kwargs...,
    )
function state_index(
    basis::PhotonBasis,
    particle::Integer,
    photon::Integer,
)
    tensor_index = state_index(basis.tensor, particle, photon)
    row = findfirst(==(tensor_index), rowvals(basis.selection))
    row === nothing &&
        throw(ArgumentError("particle/photon state lies outside the selected Ntot sector"))
    return row
end
operator_matrix(
    basis::PhotonBasis,
    opstring,
    couplings;
    sparse::Bool=false,
) = begin
    full = operator_matrix(basis.tensor, opstring, couplings; sparse=true)
    projected = basis.selection' * full * basis.selection
    sparse ? projected : Matrix(projected)
end
inplace_op!(out, basis::PhotonBasis, opstring, couplings) =
    begin
        size(out) == (length(basis), length(basis)) ||
            throw(DimensionMismatch("out must have shape (Ns,Ns)"))
        out .+= operator_matrix(basis, opstring, couplings; sparse=true)
        out
    end
