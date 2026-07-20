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

function Base.getproperty(basis::TensorBasis, name::Symbol)
    name === :N && return (
        getproperty(getfield(basis, :basis_left), :N),
        getproperty(getfield(basis, :basis_right), :N),
    )
    name === :Ns && return length(basis)
    name === :sps && return (
        getproperty(getfield(basis, :basis_left), :sps),
        getproperty(getfield(basis, :basis_right), :sps),
    )
    name === :blocks && return Dict(
        :left => getproperty(getfield(basis, :basis_left), :blocks),
        :right => getproperty(getfield(basis, :basis_right), :blocks),
    )
    name === :operators && return (
        getproperty(getfield(basis, :basis_left), :operators),
        getproperty(getfield(basis, :basis_right), :operators),
    )
    return getfield(basis, name)
end

projection_matrix(
    basis::TensorBasis,
    ::Type{T}=Float64;
    sparse::Bool=false,
) where {T<:Number} =
    kron(
        projection_matrix(basis.basis_left, T; sparse),
        projection_matrix(basis.basis_right, T; sparse),
    )

_full_projection_dimension(basis::TensorBasis) =
    _full_projection_dimension(basis.basis_left) *
    _full_projection_dimension(basis.basis_right)

function project_from(
    basis::TensorBasis,
    vector::AbstractVecOrMat;
    kwargs...,
)
    size(vector, 1) == length(basis) ||
        throw(DimensionMismatch("the first vector dimension must equal Ns"))
    sparse_projection = get(kwargs, :sparse, true)
    left_dimension = length(basis.basis_left)
    right_dimension = length(basis.basis_right)
    function project_column(column)
        coefficients = reshape(column, right_dimension, left_dimension)
        right_expanded = project_from(
            basis.basis_right,
            coefficients;
            sparse=sparse_projection,
        )
        left_expanded = project_from(
            basis.basis_left,
            transpose(right_expanded);
            sparse=sparse_projection,
        )
        return vec(transpose(left_expanded))
    end
    if vector isa AbstractVector
        return project_column(vector)
    end
    output_dimension =
        _full_projection_dimension(basis)
    isempty(axes(vector, 2)) &&
        return Matrix{eltype(vector)}(undef, output_dimension, 0)
    first_column = first(axes(vector, 2))
    first_projected =
        project_column(@view(vector[:, first_column]))
    result = Matrix{eltype(first_projected)}(
        undef,
        output_dimension,
        size(vector, 2),
    )
    result[:, first_column] = first_projected
    for column in Iterators.drop(axes(vector, 2), 1)
        result[:, column] = project_column(@view(vector[:, column]))
    end
    return result
end

get_vec(basis::TensorBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis, vector; kwargs...)
expanded_form(basis::TensorBasis, static=Any[], dynamic=Any[]) =
    (static, dynamic)

function state_index(
    basis::TensorBasis,
    left_index::Integer,
    right_index::Integer,
)
    checkbounds(1:length(basis.basis_left), left_index)
    checkbounds(1:length(basis.basis_right), right_index)
    return (left_index - 1) * length(basis.basis_right) + right_index
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
    kwargs...,
)
    return _tensor_reductions(basis, state, return_rdm)
end

function ent_entropy(
    basis::TensorBasis,
    state::AbstractVecOrMat;
    return_rdm=nothing,
    alpha::Real=1.0,
    kwargs...,
)
    if state isa AbstractVector
        coefficients = reshape(
            state,
            length(basis.basis_right),
            length(basis.basis_left),
        )
        probabilities = _schmidt_probabilities(coefficients)
        entropy = _entropy_from_probabilities(probabilities, alpha)
        result = Dict{String,Any}("Sent_A" => entropy)
        if return_rdm in (:A, "A", :both, "both")
            result["rdm_A"] = coefficients' * coefficients
        end
        if return_rdm in (:B, "B", :both, "both")
            result["Sent_B"] = entropy
            result["rdm_B"] = coefficients * coefficients'
        end
        return result
    end
    need_right = return_rdm in (:B, "B", :both, "both")
    rho_left, rho_right = if need_right
        partial_trace(basis, state; return_rdm=:both)
    else
        partial_trace(basis, state; return_rdm=:A), nothing
    end
    result = Dict{String,Any}("Sent_A" => _entropy_from_density(rho_left, alpha))
    if return_rdm in (:A, "A", :both, "both")
        result["rdm_A"] = rho_left
    end
    if need_right
        result["Sent_B"] = _entropy_from_density(rho_right, alpha)
        result["rdm_B"] = rho_right
    end
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
    length(pieces) == 2 ||
        throw(ArgumentError("tensor operator strings require one '|' separator"))
    left_op, right_op = pieces
    matrix = sparse ?
        spzeros(ComplexF64, length(basis), length(basis)) :
        zeros(ComplexF64, length(basis), length(basis))
    for coupling in couplings
        left, right = _tensor_operator_factors(
            basis,
            left_op,
            right_op,
            coupling,
            sparse,
        )
        if sparse
            matrix += kron(left, right)
        else
            matrix .+= kron(left, right)
        end
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
    pieces = split(opstring, "|"; keepempty=true)
    length(pieces) == 2 ||
        throw(ArgumentError("tensor operator strings require one '|' separator"))
    left_op, right_op = pieces
    for coupling in couplings
        left, right = _tensor_operator_factors(
            basis,
            left_op,
            right_op,
            coupling,
            true,
        )
        _accumulate_kron!(out, left, right)
    end
    return out
end

struct PhotonBasis{P<:AbstractBasis,B<:TensorBasis} <: AbstractBasis
    basis_particle::P
    basis_photon::BosonBasis1D
    tensor::B
    Nph::Int
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
    cutoff = Nph === nothing ? Int(Ntot) : Int(Nph)
    particle = basis_constructor(constructor_args...; kwargs...)
    photon = BosonBasis1D(1; sps=cutoff + 1)
    tensor = TensorBasis(particle, photon)
    return PhotonBasis(particle, photon, tensor, cutoff)
end

Base.length(basis::PhotonBasis) = length(basis.tensor)
_full_projection_dimension(basis::PhotonBasis) =
    _full_projection_dimension(basis.tensor)

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
    name === :blocks && return Dict(:Nph => getfield(basis, :Nph))
    name === :operators && return getproperty(getfield(basis, :tensor), :operators)
    return getfield(basis, name)
end

projection_matrix(
    basis::PhotonBasis,
    ::Type{T}=Float64;
    sparse::Bool=false,
) where {T<:Number} =
    projection_matrix(basis.tensor, T; sparse)
project_from(basis::PhotonBasis, vector::AbstractVecOrMat; kwargs...) =
    project_from(basis.tensor, vector; kwargs...)
get_vec(basis::PhotonBasis, vector::AbstractVecOrMat; kwargs...) =
    get_vec(basis.tensor, vector; kwargs...)
expanded_form(basis::PhotonBasis, static=Any[], dynamic=Any[]) =
    expanded_form(basis.tensor, static, dynamic)
partial_trace(basis::PhotonBasis, state::AbstractVecOrMat; kwargs...) =
    partial_trace(basis.tensor, state; kwargs...)
ent_entropy(basis::PhotonBasis, state::AbstractVecOrMat; kwargs...) =
    ent_entropy(basis.tensor, state; kwargs...)
state_index(basis::PhotonBasis, particle::Integer, photon::Integer) =
    state_index(basis.tensor, particle, photon)
operator_matrix(
    basis::PhotonBasis,
    opstring,
    couplings;
    sparse::Bool=false,
) = operator_matrix(basis.tensor, opstring, couplings; sparse)
inplace_op!(out, basis::PhotonBasis, opstring, couplings) =
    inplace_op!(out, basis.tensor, opstring, couplings)
