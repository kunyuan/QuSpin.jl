module Tools

using LinearAlgebra
using SparseArrays
using ..Basis:
    AbstractBasis,
    FixedUInt,
    get_basis_type,
    partial_trace,
    projection_matrix
using ..Operators:
    Hamiltonian,
    OperatorTerm,
    _krylov_expmv,
    _krylov_expmv_times,
    tocsc,
    toarray
import ..Basis: ent_entropy
import ..Operators: apply, evolve, set_a!

export array_to_ints, get_matvec_function, ints_to_array, kl_div, matvec
export mean_level_spacing, project_op
export ed_state_vs_time, expm_lanczos, lanczos_full, lanczos_iter, lin_comb_Q_T
export ftlm_static_iteration, ltlm_static_iteration
export diag_ensemble, ent_entropy, obs_vs_time
export evolve
export ExpmMultiplyParallel, expm_multiply_parallel
export FloquetTimeVector, get_coordinates
export Floquet
export BlockOps, block_diag_hamiltonian, block_expm
export compute_all_blocks!, update_blocks!

"""
    kl_div(p1, p2)

Kullback-Leibler divergence between strictly positive normalized discrete
probability distributions.
"""
function kl_div(p1::AbstractVector, p2::AbstractVector)
    length(p1) == length(p2) ||
        throw(DimensionMismatch("p1 and p2 must have the same length"))
    all(>(0), p1) && all(>(0), p2) ||
        throw(ArgumentError("probabilities must be strictly positive"))
    sum_p1 = sum(p1)
    sum_p2 = sum(p2)
    isapprox(sum_p1, one(sum_p1); atol=1e-13, rtol=0) ||
        throw(ArgumentError("p1 must be normalized"))
    isapprox(sum_p2, one(sum_p2); atol=1e-13, rtol=0) ||
        throw(ArgumentError("p2 must be normalized"))
    return sum(x * log(x / y) for (x, y) in zip(p1, p2))
end

"""
    mean_level_spacing(energies; verbose=true)

Mean adjacent-gap ratio of an ascending, nondegenerate energy spectrum.
"""
function mean_level_spacing(energies::AbstractVector; verbose::Bool=true)
    length(energies) >= 3 ||
        throw(ArgumentError("at least three energies are required"))
    issorted(energies) ||
        throw(ArgumentError("energies must be sorted in ascending order"))
    previous_spacing = energies[2] - energies[1]
    if iszero(previous_spacing)
        verbose && @warn "degeneracies found in energy spectrum"
        return NaN
    end
    total = zero(float(real(previous_spacing)))
    for index in 2:(length(energies) - 1)
        spacing = energies[index + 1] - energies[index]
        if iszero(spacing)
            verbose && @warn "degeneracies found in energy spectrum"
            return NaN
        end
        total += min(previous_spacing, spacing) /
            max(previous_spacing, spacing)
        previous_spacing = spacing
    end
    return total / (length(energies) - 2)
end

function _integer_width(::Type{T}) where {T<:Base.BitInteger}
    return 8 * sizeof(T)
end
_integer_width(::Type{FixedUInt{W}}) where {W} = W

"""
    ints_to_array(basis_ints, N=nothing)

Convert integer-encoded binary basis states into a row-batched, most-
significant-site-first `UInt8` matrix.
"""
function ints_to_array(basis_ints, N::Union{Nothing,Integer}=nothing)
    values = collect(basis_ints)
    isempty(values) && N === nothing &&
        throw(ArgumentError("N is required for an empty input"))
    width = if N === nothing
        T = eltype(values)
        T <: Union{Base.BitInteger,FixedUInt} ?
            _integer_width(T) :
            maximum(max(1, ndigits(BigInt(value); base=2)) for value in values)
    else
        Int(N)
    end
    width >= 0 || throw(ArgumentError("N must be nonnegative"))
    result = Matrix{UInt8}(undef, length(values), width)
    T = eltype(values)
    if T <: Union{Base.BitInteger,FixedUInt}
        for (row, encoded) in pairs(values)
            !isless(encoded, zero(encoded)) ||
                throw(ArgumentError("basis integers must be nonnegative"))
            for column in 1:width
                bit =
                    (encoded >> (width - column)) & one(encoded)
                result[row, column] = iszero(bit) ? 0x00 : 0x01
            end
        end
    else
        for (row, value) in pairs(values)
            encoded = BigInt(value)
            encoded >= 0 ||
                throw(ArgumentError("basis integers must be nonnegative"))
            for column in 1:width
                result[row, column] =
                    UInt8((encoded >> (width - column)) & 1)
            end
        end
    end
    return result
end

"""
    array_to_ints(state_array, dtype=nothing)

Convert rows of a binary state matrix to the smallest compatible basis integer
type, or to an explicitly requested `dtype`.
"""
function array_to_ints(state_array, dtype::Union{Nothing,Type}=nothing)
    states = state_array isa AbstractVector ? reshape(state_array, 1, :) : state_array
    ndims(states) == 2 || throw(ArgumentError("state_array must be one- or two-dimensional"))
    all(bit -> bit == 0 || bit == 1, states) ||
        throw(ArgumentError("state_array entries must be binary"))
    T = dtype === nothing ? get_basis_type(size(states, 2), nothing, 2) : dtype
    result = Vector{T}(undef, size(states, 1))
    native_path =
        T <: Union{Unsigned,FixedUInt} &&
        size(states, 2) <= _integer_width(T)
    if native_path
        for row in axes(states, 1)
            value = zero(T)
            for bit in @view states[row, :]
                value = (value << 1) | T(bit)
            end
            result[row] = value
        end
    else
        for row in axes(states, 1)
            value = BigInt(0)
            for bit in @view states[row, :]
                value = (value << 1) | Int(bit)
            end
            result[row] = T(value)
        end
    end
    return result
end

"""
    matvec(array, other; overwrite_out=false, out=nothing, a=1)

Compute `a * array * other`, optionally accumulating into a caller-provided
output array.
"""
function matvec(
    array,
    other;
    overwrite_out::Bool=false,
    out=nothing,
    a::Number=1,
)
    if out !== nothing &&
       applicable(
        mul!,
        out,
        array,
        other,
        a,
        overwrite_out ? zero(a) : one(a),
    )
        mul!(
            out,
            array,
            other,
            a,
            overwrite_out ? zero(a) : one(a),
        )
        return out
    end
    result = a == one(a) ? array * other : a * (array * other)
    out === nothing && return result
    axes(out) == axes(result) ||
        throw(DimensionMismatch("out and matrix-vector result must have the same axes"))
    if overwrite_out
        copyto!(out, result)
    else
        out .+= result
    end
    return out
end

get_matvec_function(array) = matvec

function _trace_product(left::AbstractMatrix, right::AbstractMatrix)
    size(left, 2) == size(right, 1) &&
        size(left, 1) == size(right, 2) ||
        throw(DimensionMismatch("trace product requires transposed dimensions"))
    T = promote_type(eltype(left), eltype(right))
    result = zero(T)
    @inbounds for column in axes(left, 2), row in axes(left, 1)
        result += left[row, column] * right[column, row]
    end
    return result
end

_projector(proj::AbstractMatrix{T}, ::Type{T}) where {T<:Number} = proj
_projector(proj::SparseMatrixCSC, ::Type{T}) where {T<:Number} =
    SparseMatrixCSC{T,Int}(proj)
_projector(proj::AbstractMatrix, ::Type{T}) where {T<:Number} =
    Matrix{T}(proj)
_projector(proj::AbstractBasis, ::Type{T}) where {T<:Number} =
    projection_matrix(proj, T; sparse=true)

_observable_matrix(observable::Hamiltonian) = tocsc(observable; time=0)

"""
    project_op(observable, projector; dtype=ComplexF64)

Project a square observable down (`P' * O * P`) or up (`P * O * P'`)
depending on which projector dimension matches the observable.
"""
function project_op(observable, proj; dtype::Type=ComplexF64)
    Obs = _observable_matrix(observable)
    ndims(Obs) == 2 && size(Obs, 1) == size(Obs, 2) ||
        throw(ArgumentError("observable must be a square matrix"))
    P = _projector(proj, dtype)
    projected = if size(Obs, 1) == size(P, 1)
        P' * (Obs * P)
    elseif size(Obs, 1) == size(P, 2)
        P * (Obs * P')
    else
        throw(DimensionMismatch("observable and projector dimensions do not match"))
    end
    return Dict("Proj_Obs" => projected)
end

"""
    ed_state_vs_time(psi, E, V, times; iterate=false)

Time-evolve a pure state or density matrix from a complete eigendecomposition.
Pure-state results use columns for time, matching QuSpin's returned array.
"""
function ed_state_vs_time(
    psi::AbstractVecOrMat,
    E::AbstractVector,
    V::AbstractMatrix,
    times;
    iterate::Bool=false,
)
    size(V, 1) == size(V, 2) || throw(ArgumentError("V must be square"))
    size(V, 1) == length(E) ||
        throw(DimensionMismatch("V and E dimensions must agree"))
    length(times) > 0 || throw(ArgumentError("times must be nonempty"))
    if psi isa AbstractVector
        length(psi) == length(E) ||
            throw(DimensionMismatch("psi and E dimensions must agree"))
        coefficients = V' * psi
        if iterate
            return (
                V * (exp.((-im * time) .* E) .* coefficients)
                for time in times
            )
        end
        time_values = collect(times)
        weights =
            exp.((-im .* E) .* transpose(time_values)) .*
            coefficients
        return V * weights
    end

    size(psi, 1) == size(psi, 2) == length(E) ||
        throw(DimensionMismatch("a mixed state must be square and match E"))
    rho_eigen = V' * psi * V
    evolved = (
        begin
            phases = exp.((-im * time) .* E)
            V * (phases .* rho_eigen .* transpose(conj.(phases))) * V'
        end
        for time in times
    )
    iterate && return evolved
    return cat(collect(evolved)...; dims=3)
end

"""
    ent_entropy(system_state, basis; chain_subsys=nothing, DM=false, ...)

Compatibility entry point for QuSpin's deprecated measurements helper. New
Julia code should prefer `ent_entropy(basis, system_state; ...)`.
"""
function ent_entropy(
    system_state::AbstractDict,
    basis::AbstractBasis;
    kwargs...,
)
    state = if haskey(system_state, "V_states") ||
               haskey(system_state, :V_states)
        get(system_state, "V_states", get(system_state, :V_states, nothing))
    elseif (
        haskey(system_state, "V_rho") ||
        haskey(system_state, :V_rho)
    ) && (
        haskey(system_state, "rho_d") ||
        haskey(system_state, :rho_d)
    )
        vectors =
            get(system_state, "V_rho", get(system_state, :V_rho, nothing))
        probabilities =
            get(system_state, "rho_d", get(system_state, :rho_d, nothing))
        vectors * Diagonal(probabilities) * vectors'
    else
        throw(ArgumentError(
            "state dictionary requires V_states or both V_rho and rho_d",
        ))
    end
    return ent_entropy(
        state,
        basis;
        enforce_pure=(
            haskey(system_state, "V_states") ||
            haskey(system_state, :V_states)
        ),
        kwargs...,
    )
end

function ent_entropy(
    system_state::AbstractVecOrMat,
    basis::AbstractBasis;
    chain_subsys=nothing,
    DM=false,
    svd_return_vec=(false, false, false),
    alpha::Real=1.0,
    kwargs...,
)
    requested = DM in (:both, "both") ? :both :
                DM in (:chain_subsys, "chain_subsys", :A, "A") ? :A :
                DM in (:other_subsys, "other_subsys", :B, "B") ? :B :
                nothing
    result = ent_entropy(
        basis,
        system_state;
        sub_sys_A=chain_subsys,
        return_rdm=requested,
        alpha,
        kwargs...,
    )
    result["Sent"] = result["Sent_A"]
    haskey(result, "rdm_A") && (result["DM_chain_subsys"] = result["rdm_A"])
    haskey(result, "rdm_B") && (result["DM_other_subsys"] = result["rdm_B"])
    return result
end

_observable_matrix(observable::AbstractMatrix) = observable
_observable_matrix(observable) = Matrix(observable)
_observable_matrix(observable, time) = observable isa Function ?
    _observable_matrix(observable(time)) :
    _observable_matrix(observable)

function _pure_expectation(
    state::AbstractVector,
    matrix::AbstractMatrix,
    scratch=nothing,
)
    T = promote_type(eltype(state), eltype(matrix))
    work = scratch === nothing || length(scratch) != length(state) ||
           eltype(scratch) != T ?
        similar(state, T) :
        scratch
    if applicable(mul!, work, matrix, state)
        mul!(work, matrix, state)
        return dot(state, work), work
    end
    applied = matrix * state
    return dot(state, applied), work
end

function _obs_vs_time_stream(
    psi_t,
    time_values,
    observables::AbstractDict;
    verbose::Bool=false,
)
    results = Dict{Any,Any}()
    scratch = Dict{Any,Any}()
    append_result! = function (key, value)
        if !haskey(results, key)
            results[key] = typeof(value)[value]
            return
        end
        values = results[key]
        T = promote_type(eltype(values), typeof(value))
        if T == eltype(values)
            push!(values, value)
        else
            promoted = T[values...]
            push!(promoted, value)
            results[key] = promoted
        end
        return
    end
    iteration = iterate(psi_t)
    for (index, time) in pairs(time_values)
        iteration === nothing &&
            throw(DimensionMismatch("state iterator ended before times"))
        state, iterator_state = iteration
        if state isa AbstractVector
            for (key, observable) in observables
                matrix = _observable_matrix(observable, time)
                value, work = _pure_expectation(
                    state,
                    matrix,
                    get(scratch, key, nothing),
                )
                scratch[key] = work
                append_result!(key, value)
            end
        elseif state isa AbstractMatrix
            for (key, observable) in observables
                value = _trace_product(
                    state,
                    _observable_matrix(observable, time),
                )
                append_result!(key, value)
            end
        else
            throw(ArgumentError("state iterator must yield vectors or matrices"))
        end
        verbose && @info "obs_vs_time measured" time
        iteration = iterate(psi_t, iterator_state)
    end
    iteration === nothing ||
        throw(DimensionMismatch("state iterator contains more states than times"))
    return results
end

"""
    obs_vs_time(psi_t, times, observables; ...)

Expectation values of static observables for pure-state columns, density-matrix
time slices, a state iterator, or `(psi, E, V)` eigensystem input.
"""
function obs_vs_time(
    psi_t,
    times,
    observables::AbstractDict;
    return_state::Bool=false,
    Sent_args::AbstractDict=Dict(),
    enforce_pure::Bool=false,
    verbose::Bool=false,
)
    time_values = collect(times)
    isempty(time_values) && throw(ArgumentError("times must be nonempty"))
    if !(psi_t isa Tuple) &&
       !(psi_t isa AbstractArray) &&
       !return_state &&
       isempty(Sent_args)
        return _obs_vs_time_stream(
            psi_t,
            time_values,
            observables;
            verbose,
        )
    end
    states = if psi_t isa Tuple
        length(psi_t) == 3 ||
            throw(ArgumentError("tuple input must be (psi, E, V)"))
        psi, E, V = psi_t
        ed_state_vs_time(psi, E, V, time_values)
    elseif psi_t isa AbstractArray
        psi_t
    else
        collected = collect(psi_t)
        isempty(collected) && throw(ArgumentError("state iterator must be nonempty"))
        first(collected) isa AbstractVector ?
            reduce(hcat, collected) :
            cat(collected...; dims=3)
    end

    results = Dict{Any,Any}()
    if ndims(states) == 2 && (enforce_pure || size(states, 2) == length(time_values))
        size(states, 2) == length(time_values) ||
            throw(DimensionMismatch("one pure-state column is required per time"))
        for (key, observable) in observables
            if observable isa Function
                work = nothing
                values = Vector{Any}(undef, size(states, 2))
                for index in axes(states, 2)
                    values[index], work = _pure_expectation(
                        @view(states[:, index]),
                        _observable_matrix(observable, time_values[index]),
                        work,
                    )
                end
                value_type = foldl(
                    promote_type,
                    (typeof(value) for value in values),
                )
                results[key] = value_type[value for value in values]
            else
                matrix = _observable_matrix(observable)
                applied = matrix * states
                results[key] = [
                    dot(
                        @view(states[:, index]),
                        @view(applied[:, index]),
                    )
                    for index in axes(states, 2)
                ]
            end
        end
    elseif ndims(states) == 3
        size(states, 3) == length(time_values) ||
            throw(DimensionMismatch("one density-matrix slice is required per time"))
        for (key, observable) in observables
            results[key] = [
                begin
                    matrix = _observable_matrix(observable, time_values[index])
                    _trace_product(@view(states[:, :, index]), matrix)
                end
                for index in axes(states, 3)
            ]
        end
    else
        throw(ArgumentError("states must be pure-state columns or density-matrix slices"))
    end

    if !isempty(Sent_args)
        basis = get(Sent_args, :basis, get(Sent_args, "basis", nothing))
        basis isa AbstractBasis ||
            throw(ArgumentError("Sent_args must contain a basis"))
        entropy_kwargs = Dict(
            Symbol(key) => value
            for (key, value) in Sent_args
            if key ∉ (:basis, "basis")
        )
        entropy_rows = if ndims(states) == 2
            [
                ent_entropy(
                    basis,
                    @view(states[:, index]);
                    entropy_kwargs...,
                )
                for index in axes(states, 2)
            ]
        else
            [
                ent_entropy(
                    basis,
                    @view(states[:, :, index]);
                    entropy_kwargs...,
                )
                for index in axes(states, 3)
            ]
        end
        entropy_keys = keys(first(entropy_rows))
        results["Sent_time"] = Dict(
            key => [entry[key] for entry in entropy_rows]
            for key in entropy_keys
        )
        return_state = true
    end
    return_state && (results["psi_t"] = states)
    return results
end

_state_label(state::AbstractVector) = "pure"
_state_label(state::AbstractMatrix) = "DM"

function _diagonal_probabilities(state::AbstractVector, eigenvectors)
    length(state) == size(eigenvectors, 1) ||
        throw(DimensionMismatch("state and V2 dimensions do not match"))
    return abs2.(eigenvectors' * state), "pure"
end

function _diagonal_probabilities(state::AbstractMatrix, eigenvectors)
    size(state) == (size(eigenvectors, 1), size(eigenvectors, 1)) ||
        throw(DimensionMismatch("density matrix and V2 dimensions do not match"))
    probabilities = _diagonal_in_basis(state, eigenvectors)
    return probabilities, "DM"
end

function _diagonal_in_basis(operator, eigenvectors::AbstractMatrix)
    transformed = operator * eigenvectors
    T = typeof(real(dot(
        @view(eigenvectors[:, first(axes(eigenvectors, 2))]),
        @view(transformed[:, first(axes(transformed, 2))]),
    )))
    diagonal = Vector{T}(undef, size(eigenvectors, 2))
    for column in axes(eigenvectors, 2)
        diagonal[column] = real(dot(
            @view(eigenvectors[:, column]),
            @view(transformed[:, column]),
        ))
    end
    return diagonal
end

function _diagonal_of_square(matrix::AbstractMatrix)
    size(matrix, 1) == size(matrix, 2) ||
        throw(ArgumentError("matrix must be square"))
    T = promote_type(eltype(matrix), typeof(zero(eltype(matrix))))
    result = Vector{T}(undef, size(matrix, 1))
    @inbounds for row in axes(matrix, 1)
        value = zero(T)
        for index in axes(matrix, 2)
            value += matrix[row, index] * matrix[index, row]
        end
        result[row] = value
    end
    return result
end

function _weighted_reduced_density(
    basis::AbstractBasis,
    eigenvectors::AbstractMatrix,
    probabilities::AbstractVector,
    subsystem,
)
    size(eigenvectors, 2) == length(probabilities) ||
        throw(DimensionMismatch("one probability is required per eigenvector"))
    first_reduction = partial_trace(
        basis,
        @view(eigenvectors[:, first(axes(eigenvectors, 2))]);
        sub_sys_A=subsystem,
        return_rdm=:A,
    )
    T = promote_type(eltype(first_reduction), eltype(probabilities))
    reduced = zeros(T, size(first_reduction))
    reduced .+= probabilities[1] .* first_reduction
    for column in Iterators.drop(axes(eigenvectors, 2), 1)
        reduction = partial_trace(
            basis,
            @view(eigenvectors[:, column]);
            sub_sys_A=subsystem,
            return_rdm=:A,
        )
        reduced .+= probabilities[column] .* reduction
    end
    return reduced
end

function _entropy_from_density_matrix(rho::AbstractMatrix, alpha::Real)
    probabilities = real.(eigvals(Hermitian((rho + rho') / 2)))
    tolerance = 100eps(float(real(one(eltype(probabilities)))))
    probabilities = clamp.(probabilities, zero(eltype(probabilities)), Inf)
    probabilities = probabilities[probabilities .> tolerance]
    isempty(probabilities) && return zero(eltype(probabilities))
    if abs(alpha - 1) <= sqrt(eps(float(alpha)))
        return -sum(value * log(value) for value in probabilities)
    end
    return log(sum(value^alpha for value in probabilities)) / (1 - alpha)
end

function _diagonal_entropy(probabilities, alpha::Real)
    columns = probabilities isa AbstractVector ?
        (probabilities,) :
        eachcol(probabilities)
    T = typeof(float(real(zero(eltype(probabilities)))))
    values = Vector{T}(undef, probabilities isa AbstractVector ? 1 : size(probabilities, 2))
    for (index, column) in enumerate(columns)
        values[index] = if abs(alpha - 1) <= sqrt(eps(float(alpha)))
            -sum(
                probability > 0 ? probability * log(probability) : zero(T)
                for probability in column
            )
        else
            log(sum(
                probability > 0 ? probability^alpha : zero(T)
                for probability in column
            )) / (1 - alpha)
        end
    end
    return values
end

function _subsystem_normalization(basis::AbstractBasis, subsystem)
    if subsystem !== nothing
        return max(1, length(subsystem))
    end
    hasproperty(basis, :L) || return 1
    return max(1, fld(Int(getproperty(basis, :L)), 2))
end

function _diagonal_probabilities(state::AbstractDict, eigenvectors)
    V1 = _dict_get(state, :V1)
    E1 = _dict_get(state, :E1)
    parameters = _dict_get(state, :f_args)
    V1 === nothing && throw(ArgumentError("mixed state requires V1"))
    E1 === nothing && throw(ArgumentError("mixed state requires E1"))
    parameters === nothing && throw(ArgumentError("mixed state requires f_args"))
    distribution = _dict_get(
        state,
        :f,
        (energies, beta) -> exp.(-beta .* (energies .- first(energies))),
    )
    parameter_values = collect(parameters)
    !isempty(parameter_values) && first(parameter_values) isa AbstractArray &&
        (parameter_values = collect(first(parameter_values)))
    isempty(parameter_values) &&
        throw(ArgumentError("f_args must contain at least one parameter"))
    overlaps = abs2.(eigenvectors' * V1)
    first_values = distribution(E1, first(parameter_values))
    normalized = first_values ./ sum(first_values)
    weights = Matrix{eltype(normalized)}(
        undef,
        length(normalized),
        length(parameter_values),
    )
    copyto!(@view(weights[:, 1]), normalized)
    for index in 2:length(parameter_values)
        values = distribution(E1, parameter_values[index])
        copyto!(@view(weights[:, index]), values ./ sum(values))
    end
    label = _dict_has(state, :f) ? "mixed" : "thermal"
    return overlaps * weights, label
end

_result_shape(value::Number) = value
_result_shape(value::AbstractArray) = length(value) == 1 ? only(value) : vec(value)

"""
    diag_ensemble(N, system_state, E2, V2; ...)

Infinite-time diagonal-ensemble probabilities, observables, fluctuations and
Rényi entropy in the post-quench eigenbasis.
"""
function diag_ensemble(
    N::Integer,
    system_state,
    E2,
    V2;
    density::Bool=true,
    alpha::Real=1.0,
    rho_d::Bool=false,
    Obs=nothing,
    delta_t_Obs::Bool=false,
    delta_q_Obs::Bool=false,
    Sd_Renyi::Bool=false,
    Srdm_Renyi::Bool=false,
    Srdm_args::AbstractDict=Dict(),
)
    N > 0 || throw(ArgumentError("N must be positive"))
    alpha >= 0 || throw(ArgumentError("alpha must be nonnegative"))
    energies = sort!(collect(E2))
    length(energies) == size(V2, 2) ||
        throw(DimensionMismatch("E2 and V2 dimensions do not match"))
    tolerance = 1000eps(float(real(one(eltype(energies)))))
    for index in 2:length(energies)
        energies[index] - energies[index - 1] > tolerance ||
            throw(ArgumentError("E2 must be nondegenerate"))
    end
    probabilities, label = _diagonal_probabilities(system_state, V2)
    probabilities =
        map(
            probability -> max(
                real(probability),
                zero(real(probability)),
            ),
            probabilities,
        )
    requested_observable = !(Obs === nothing || Obs === false)
    (delta_t_Obs || delta_q_Obs) && !requested_observable &&
        throw(ArgumentError("observable fluctuations require Obs"))

    result = Dict{String,Any}()
    divisor = density ? N : 1
    observable_values = nothing
    temporal_variance = nothing
    if requested_observable
        observable = _observable_matrix(Obs)
        size(observable) == (size(V2, 1), size(V2, 1)) ||
            throw(DimensionMismatch("Obs and V2 dimensions do not match"))
        eigen_observable = if delta_t_Obs || delta_q_Obs
            V2' * (observable * V2)
        else
            nothing
        end
        diagonal_values = eigen_observable === nothing ?
            _diagonal_in_basis(observable, V2) :
            real.(diag(eigen_observable))
        observable_values = _result_shape(transpose(diagonal_values) * probabilities)
        result["Obs_$label"] = observable_values / divisor

        if delta_t_Obs || delta_q_Obs
            off_diagonal = abs2.(eigen_observable)
            off_diagonal[diagind(off_diagonal)] .= 0
            temporal_variance = _result_shape(
                sum(
                    probabilities .* (off_diagonal * probabilities);
                    dims=1,
                ),
            )
            result["delta_t_Obs_$label"] =
                sqrt.(max.(temporal_variance, 0)) / divisor
            if delta_q_Obs
                squared_diagonal = real.(_diagonal_of_square(eigen_observable))
                total_second = _result_shape(
                    transpose(squared_diagonal) * probabilities,
                )
                quantum_variance =
                    total_second .- temporal_variance .- observable_values .^ 2
                result["delta_q_Obs_$label"] =
                    sqrt.(max.(quantum_variance, 0)) / divisor
            end
        end
    end

    if Sd_Renyi
        entropy = _diagonal_entropy(probabilities, alpha)
        key = alpha == 1 ? "Sd_$label" : "Sd_Renyi_$label"
        result[key] = _result_shape(entropy) / divisor
    end

    if Srdm_Renyi
        ndims(probabilities) == 1 ||
            throw(ArgumentError("Srdm_Renyi currently requires one diagonal ensemble"))
        basis = _dict_get(Srdm_args, :basis)
        basis isa AbstractBasis || throw(ArgumentError("Srdm_args requires a basis"))
        subsystem = _dict_get(
            Srdm_args,
            :sub_sys_A,
            _dict_get(Srdm_args, :chain_subsys, nothing),
        )
        reduced_density = _weighted_reduced_density(
            basis,
            V2,
            vec(probabilities),
            subsystem,
        )
        entropy = _entropy_from_density_matrix(reduced_density, alpha) /
            _subsystem_normalization(basis, subsystem)
        key = alpha == 1 ? "Srdm_$label" : "Srdm_Renyi_$label"
        result[key] = entropy
    end

    rho_d && (result["rho_d"] = probabilities)
    return result
end

"""
    evolve(v0, t0, times, f; solver_name=:dop853, f_params=(),
           max_step=0.01, iterate=false, ...)

Integrate a user-defined first-order ODE. The default error-controlled
Dormand-Prince 5(4) path accepts `solver_name=:dop853`, `:dopri5`, or `:rk45`;
the fixed-step fourth-order path remains available as `solver_name=:rk4`.
The derivative may be out-of-place, `f(t, state, f_params...)`, or in-place,
`f(destination, t, state, f_params...)`.
"""
mutable struct _RK4EvolutionIterator{S,TV,F,P,R}
    state::S
    targets::TV
    f::F
    f_params::P
    current::R
    max_step::R
    imag_time::Bool
    verbose::Bool
    index::Int
    buffers::NTuple{5,S}
end

Base.IteratorSize(::Type{<:_RK4EvolutionIterator}) = Base.SizeUnknown()
Base.eltype(::Type{_RK4EvolutionIterator{S}}) where {S} = S

function _copy_derivative!(destination, derivative)
    axes(destination) == axes(derivative) ||
        throw(DimensionMismatch("ODE derivative must match the state axes"))
    copyto!(destination, derivative)
    return destination
end

function _evaluate_derivative!(destination, f, time, state, f_params)
    if applicable(f, destination, time, state, f_params...)
        returned = f(destination, time, state, f_params...)
        if returned !== nothing && returned !== destination
            _copy_derivative!(destination, returned)
        end
        return destination
    end
    return _copy_derivative!(
        destination,
        f(time, state, f_params...),
    )
end

function Base.iterate(iterator::_RK4EvolutionIterator, iteration_state=nothing)
    iterator.index > length(iterator.targets) && return nothing
    target = iterator.targets[iterator.index]
    target >= iterator.current ||
        throw(ArgumentError("times must not precede the current integration time"))
    interval = target - iterator.current
    state = iterator.state
    k1, k2, k3, k4, temporary = iterator.buffers
    if !iszero(interval)
        steps = max(1, ceil(Int, abs(interval) / iterator.max_step))
        step = interval / steps
        for _ in 1:steps
            _evaluate_derivative!(
                k1,
                iterator.f,
                iterator.current,
                state,
                iterator.f_params,
            )
            @. temporary = state + (step / 2) * k1
            _evaluate_derivative!(
                k2,
                iterator.f,
                iterator.current + step / 2,
                temporary,
                iterator.f_params,
            )
            @. temporary = state + (step / 2) * k2
            _evaluate_derivative!(
                k3,
                iterator.f,
                iterator.current + step / 2,
                temporary,
                iterator.f_params,
            )
            @. temporary = state + step * k3
            _evaluate_derivative!(
                k4,
                iterator.f,
                iterator.current + step,
                temporary,
                iterator.f_params,
            )
            @. state +=
                (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            iterator.current += step
        end
    end
    iterator.imag_time && (state ./= norm(state))
    iterator.verbose && @info "evolve integrated" time=target
    iterator.index += 1
    return copy(state), nothing
end

mutable struct _RK45EvolutionIterator{S,TV,F,P,R}
    state::S
    targets::TV
    f::F
    f_params::P
    current::R
    next_step::R
    max_step::R
    rtol::R
    atol::R
    imag_time::Bool
    verbose::Bool
    index::Int
    buffers::NTuple{10,S}
end

Base.IteratorSize(::Type{<:_RK45EvolutionIterator}) = Base.SizeUnknown()
Base.eltype(::Type{_RK45EvolutionIterator{S}}) where {S} = S

function _rk45_trial_step!(iterator::_RK45EvolutionIterator, step)
    state = iterator.state
    k1, k2, k3, k4, k5, k6, k7, temporary, fifth, fourth =
        iterator.buffers
    time = iterator.current
    parameters = iterator.f_params
    f = iterator.f

    _evaluate_derivative!(k1, f, time, state, parameters)
    @. temporary = state + step * (1 / 5) * k1
    _evaluate_derivative!(k2, f, time + step * (1 / 5), temporary, parameters)
    @. temporary =
        state + step * ((3 / 40) * k1 + (9 / 40) * k2)
    _evaluate_derivative!(k3, f, time + step * (3 / 10), temporary, parameters)
    @. temporary =
        state +
        step * ((44 / 45) * k1 - (56 / 15) * k2 + (32 / 9) * k3)
    _evaluate_derivative!(k4, f, time + step * (4 / 5), temporary, parameters)
    @. temporary =
        state +
        step *
        (
            (19372 / 6561) * k1 -
            (25360 / 2187) * k2 +
            (64448 / 6561) * k3 -
            (212 / 729) * k4
        )
    _evaluate_derivative!(k5, f, time + step * (8 / 9), temporary, parameters)
    @. temporary =
        state +
        step *
        (
            (9017 / 3168) * k1 -
            (355 / 33) * k2 +
            (46732 / 5247) * k3 +
            (49 / 176) * k4 -
            (5103 / 18656) * k5
        )
    _evaluate_derivative!(k6, f, time + step, temporary, parameters)
    @. fifth =
        state +
        step *
        (
            (35 / 384) * k1 +
            (500 / 1113) * k3 +
            (125 / 192) * k4 -
            (2187 / 6784) * k5 +
            (11 / 84) * k6
        )
    _evaluate_derivative!(k7, f, time + step, fifth, parameters)
    @. fourth =
        state +
        step *
        (
            (5179 / 57600) * k1 +
            (7571 / 16695) * k3 +
            (393 / 640) * k4 -
            (92097 / 339200) * k5 +
            (187 / 2100) * k6 +
            (1 / 40) * k7
        )

    error = zero(iterator.rtol)
    @inbounds for index in eachindex(state)
        scale =
            iterator.atol +
            iterator.rtol * max(abs(state[index]), abs(fifth[index]))
        error = max(
            error,
            convert(
                typeof(error),
                abs(fifth[index] - fourth[index]) / scale,
            ),
        )
    end
    return error
end

function Base.iterate(iterator::_RK45EvolutionIterator, iteration_state=nothing)
    iterator.index > length(iterator.targets) && return nothing
    target = iterator.targets[iterator.index]
    target >= iterator.current ||
        throw(ArgumentError("times must not precede the current integration time"))
    while iterator.current < target
        step = min(iterator.next_step, iterator.max_step, target - iterator.current)
        step > eps(max(abs(iterator.current), one(iterator.current))) ||
            throw(ErrorException("adaptive evolution step underflow"))
        error = _rk45_trial_step!(iterator, step)
        if error <= one(error)
            copyto!(iterator.state, iterator.buffers[9])
            iterator.current += step
        end
        factor = iszero(error) ?
            convert(typeof(step), 5) :
            clamp(
                convert(typeof(step), 0.9) * error^(-one(error) / 5),
                convert(typeof(step), 0.2),
                convert(typeof(step), 5),
            )
        iterator.next_step = min(iterator.max_step, step * factor)
    end
    if iterator.imag_time
        iterator.state ./= norm(iterator.state)
    end
    iterator.verbose && @info "evolve integrated" time=target
    iterator.index += 1
    return copy(iterator.state), nothing
end

function _rk4_iterator(
    v0::AbstractArray,
    t0::Real,
    targets,
    f,
    f_params,
    max_step::Real,
    imag_time::Bool,
    verbose::Bool,
    real_valued::Bool,
)
    base_type =
        promote_type(eltype(v0), typeof(float(t0)), typeof(float(max_step)))
    T = real_valued ?
        typeof(real(zero(base_type))) :
        typeof(complex(zero(base_type)))
    state = similar(v0, T)
    copyto!(state, v0)
    buffers = ntuple(_ -> similar(state), 5)
    R = promote_type(typeof(float(t0)), eltype(targets), typeof(float(max_step)))
    return _RK4EvolutionIterator(
        state,
        targets,
        f,
        f_params,
        convert(R, t0),
        convert(R, max_step),
        imag_time,
        verbose,
        1,
        buffers,
    )
end

function _rk45_iterator(
    v0::AbstractArray,
    t0::Real,
    targets,
    f,
    f_params,
    max_step::Real,
    rtol::Real,
    atol::Real,
    imag_time::Bool,
    verbose::Bool,
    real_valued::Bool,
)
    base_type = promote_type(
        eltype(v0),
        typeof(float(t0)),
        typeof(float(max_step)),
        typeof(float(rtol)),
        typeof(float(atol)),
    )
    T = real_valued ?
        typeof(real(zero(base_type))) :
        typeof(complex(zero(base_type)))
    state = similar(v0, T)
    copyto!(state, v0)
    buffers = ntuple(_ -> similar(state), 10)
    R = promote_type(
        typeof(float(t0)),
        eltype(targets),
        typeof(float(max_step)),
        typeof(float(rtol)),
        typeof(float(atol)),
    )
    return _RK45EvolutionIterator(
        state,
        targets,
        f,
        f_params,
        convert(R, t0),
        convert(R, max_step),
        convert(R, max_step),
        convert(R, rtol),
        convert(R, atol),
        imag_time,
        verbose,
        1,
        buffers,
    )
end

function _stacked_real_initial(state::AbstractArray)
    flattened = vec(state)
    R = typeof(real(zero(eltype(flattened))))
    return R[real.(flattened)...; imag.(flattened)...]
end

function _unstack_complex_state(state::AbstractVector, shape)
    iseven(length(state)) ||
        throw(DimensionMismatch("stacked real state must have even length"))
    count = length(state) ÷ 2
    complex_state =
        complex.(@view(state[1:count]), @view(state[(count + 1):end]))
    restored = reshape(complex_state, shape)
    return length(shape) == 1 ? vec(restored) : restored
end

function evolve(
    v0::AbstractArray,
    t0::Real,
    times,
    f;
    solver_name=:dop853,
    real::Bool=false,
    stack_state::Bool=false,
    verbose::Bool=false,
    imag_time::Bool=false,
    iterate::Bool=false,
    f_params=(),
    max_step::Real=0.01,
    kwargs...,
)
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    solver = Symbol(lowercase(String(solver_name)))
    solver in (
        :rk4,
        :dop853,
        :dopri5,
        :rk45,
        :vode,
        :zvode,
        :lsoda,
    ) ||
        throw(ArgumentError(
            "unsupported ODE solver name",
        ))
    if stack_state
        imag_time && throw(ArgumentError(
            "imag_time is incompatible with stack_state",
        ))
        internal = evolve(
            _stacked_real_initial(v0),
            t0,
            times,
            f;
            solver_name=solver,
            real=true,
            stack_state=false,
            verbose,
            imag_time=false,
            iterate,
            f_params,
            max_step,
            kwargs...,
        )
        if iterate
            return (
                _unstack_complex_state(state, size(v0))
                for state in internal
            )
        end
        if times isa Number
            return _unstack_complex_state(internal, size(v0))
        end
        restored = [
            _unstack_complex_state(
                @view(internal[:, index]),
                size(v0),
            )
            for index in axes(internal, 2)
        ]
        return v0 isa AbstractVector ?
            reduce(hcat, restored) :
            cat(restored...; dims=ndims(v0) + 1)
    end
    supported_keywords = solver === :rk4 ? () : (:rtol, :atol, :tol)
    unsupported = setdiff(keys(kwargs), supported_keywords)
    isempty(unsupported) ||
        throw(ArgumentError("unsupported ODE solver keyword(s): $unsupported"))
    scalar_target = times isa Number
    targets = scalar_target ? [times] : collect(times)
    isempty(targets) && throw(ArgumentError("times must be nonempty"))
    issorted(targets) || throw(ArgumentError("times must be sorted"))
    targets[1] >= t0 ||
        throw(ArgumentError("times must not precede t0"))
    iterator = if solver === :rk4
        _rk4_iterator(
            v0,
            t0,
            targets,
            f,
            f_params,
            max_step,
            imag_time,
            verbose,
            real,
        )
    else
        rtol = get(kwargs, :rtol, get(kwargs, :tol, 1e-9))
        atol = get(kwargs, :atol, 1e-11)
        rtol > 0 || throw(ArgumentError("rtol must be positive"))
        atol > 0 || throw(ArgumentError("atol must be positive"))
        _rk45_iterator(
            v0,
            t0,
            targets,
            f,
            f_params,
            max_step,
            rtol,
            atol,
            imag_time,
            verbose,
            real,
        )
    end
    iterate && return iterator
    outputs = collect(iterator)
    scalar_target && return first(outputs)
    first(outputs) isa AbstractVector && return reduce(hcat, outputs)
    return cat(outputs...; dims=ndims(first(outputs)) + 1)
end

"""
    ExpmMultiplyParallel(A, a=1; dtype=nothing, copy=false)

Native Julia representation of the linear action `exp(a*A) * v`. Sparse and
matrix-free inputs use the native Krylov exponential-action kernel without
materializing `exp(a*A)`.
"""
mutable struct ExpmMultiplyParallel{M,T<:Number}
    A::M
    a::T
    dtype::Type{T}
end

function ExpmMultiplyParallel(
    A,
    a::Number=1.0;
    dtype::Union{Nothing,Type}=nothing,
    copy::Bool=false,
)
    ndims(A) == 2 && size(A, 1) == size(A, 2) ||
        throw(ArgumentError("A must be a square matrix"))
    requested_type =
        dtype === nothing ?
        promote_type(eltype(A), typeof(a), Float64) :
        dtype
    requested_type <: Number || throw(ArgumentError("dtype must be numeric"))
    T = typeof(complex(zero(requested_type)))
    stored = copy ? Base.copy(A) : A
    return ExpmMultiplyParallel{typeof(stored),T}(
        stored,
        convert(T, a),
        T,
    )
end

const expm_multiply_parallel = ExpmMultiplyParallel

"""
    set_a!(operator::ExpmMultiplyParallel, a; dtype=nothing)

Update the scalar multiplying the exponential generator.
"""
function set_a!(
    operator::ExpmMultiplyParallel{M,T},
    a::Number;
    dtype::Union{Nothing,Type}=nothing,
) where {M,T}
    if dtype !== nothing
        requested_type = typeof(complex(zero(dtype)))
        requested_type == T ||
            throw(ArgumentError(
                "changing the exponential dtype requires constructing a new operator",
            ))
    end
    operator.a = convert(T, a)
    return operator
end

"""
    apply(operator::ExpmMultiplyParallel, v; work_array=nothing,
          overwrite_v=false, tol=nothing)

Apply the matrix exponential to one vector or a batch of column vectors.
"""
function apply(
    operator::ExpmMultiplyParallel{M,T},
    v::AbstractVecOrMat;
    work_array=nothing,
    overwrite_v::Bool=false,
    tol=nothing,
) where {M,T}
    size(v, 1) == size(operator.A, 2) ||
        throw(DimensionMismatch("A and v dimensions do not match"))
    if work_array !== nothing
        length(work_array) == 2length(v) ||
            throw(DimensionMismatch("work_array must contain twice as many elements as v"))
        eltype(work_array) == T ||
            throw(ArgumentError("work_array must have element type $T"))
    end
    input = if work_array === nothing
        MatrixOrVector(v, T)
    else
        reshaped = reshape(@view(work_array[1:length(v)]), size(v))
        copyto!(reshaped, v)
        reshaped
    end
    result = _krylov_expmv(
        operator.A,
        input,
        operator.a;
        tol=tol === nothing ? 1e-12 : tol,
    )
    output = if work_array === nothing
        result
    else
        reshaped = reshape(
            @view(work_array[(length(v) + 1):(2length(v))]),
            size(v),
        )
        copyto!(reshaped, result)
        reshaped
    end
    overwrite_v || return output
    eltype(v) == eltype(output) ||
        throw(ArgumentError("overwrite_v requires an input with the output dtype"))
    copyto!(v, output)
    return v
end

MatrixOrVector(v::AbstractVector{T}, ::Type{T}) where {T} = v
MatrixOrVector(v::AbstractMatrix{T}, ::Type{T}) where {T} = v
MatrixOrVector(v::AbstractVector, ::Type{T}) where {T} = Vector{T}(v)
MatrixOrVector(v::AbstractMatrix, ::Type{T}) where {T} = Matrix{T}(v)

Base.:*(operator::ExpmMultiplyParallel, v::AbstractVecOrMat) = apply(operator, v)

struct StroboscopicTimes{T}
    inds::Vector{Int}
    vals::Vector{T}
end

struct FloquetTimeSegment{T,V<:AbstractVector{T}}
    vals::V
    len::Int
    shape::Tuple{Int}
    i::T
    f::T
    tot::T
    strobo::StroboscopicTimes{T}
end

function _time_segment(values::AbstractVector{T}, stride::Int) where {T}
    vals = values
    isempty(vals) && return nothing
    inds = collect(1:stride:length(vals))
    last(inds) == length(vals) || push!(inds, length(vals))
    return FloquetTimeSegment(
        vals,
        length(vals),
        size(vals),
        first(vals),
        last(vals),
        last(vals) - first(vals),
        StroboscopicTimes(inds, vals[inds]),
    )
end

"""
    FloquetTimeVector(Omega, N_const; len_T=100, N_up=0, N_down=0)

Uniform time grid that lands exactly on every stroboscopic boundary.  Indices
and coordinates use Julia's one-based convention.
"""
struct FloquetTimeVector{T}
    N::Int
    len_T::Int
    T::T
    vals::Vector{T}
    len::Int
    shape::Tuple{Int}
    dt::T
    i::T
    f::T
    tot::T
    strobo::StroboscopicTimes{T}
    up::Union{Nothing,FloquetTimeSegment{T}}
    constant::FloquetTimeSegment{T}
    down::Union{Nothing,FloquetTimeSegment{T}}
end

function FloquetTimeVector(
    Omega::Real,
    N_const::Integer;
    len_T::Integer=100,
    N_up::Integer=0,
    N_down::Integer=0,
)
    Omega != 0 || throw(ArgumentError("Omega must be nonzero"))
    N_const >= 0 && N_up >= 0 && N_down >= 0 ||
        throw(ArgumentError("period counts must be nonnegative"))
    len_T > 0 || throw(ArgumentError("len_T must be positive"))
    N = Int(N_up + N_const + N_down)
    N > 0 || throw(ArgumentError("the time vector must contain at least one period"))
    T = 2π / float(Omega)
    vals = collect(range(-N_up * T, (N_const + N_down) * T; length=N * len_T + 1))
    inds = collect(1:Int(len_T):length(vals))
    strobo = StroboscopicTimes(inds, vals[inds])

    up_end = inds[Int(N_up) + 1]
    const_end = inds[Int(N_up + N_const) + 1]
    up = N_up > 0 ? _time_segment(@view(vals[1:(up_end - 1)]), Int(len_T)) : nothing
    constant = _time_segment(@view(vals[up_end:const_end]), Int(len_T))
    down = N_down > 0 ?
        _time_segment(@view(vals[(const_end + 1):end]), Int(len_T)) :
        nothing

    return FloquetTimeVector(
        N,
        Int(len_T),
        T,
        vals,
        length(vals),
        size(vals),
        T / len_T,
        first(vals),
        last(vals),
        last(vals) - first(vals),
        strobo,
        up,
        constant,
        down,
    )
end

Base.length(times::FloquetTimeVector) = times.len
Base.size(times::FloquetTimeVector) = times.shape
Base.iterate(times::FloquetTimeVector, state...) = iterate(times.vals, state...)
Base.getindex(times::FloquetTimeVector, index...) = getindex(times.vals, index...)
Base.:*(times::FloquetTimeVector, value::Number) = times.vals * value
Base.:/(times::FloquetTimeVector, value::Number) = times.vals / value

function Base.getproperty(times::FloquetTimeVector, name::Symbol)
    name === :const && return getfield(times, :constant)
    return getfield(times, name)
end

"""
    get_coordinates(times, index)

Return the one-based `(period, offset)` coordinates for a one-based flat index.
"""
function get_coordinates(times::FloquetTimeVector, index::Integer)
    checkbounds(times.vals, index)
    zero_based = Int(index) - 1
    return (fld(zero_based, times.len_T) + 1, mod(zero_based, times.len_T) + 1)
end

"""
    Floquet(evolution; HF=false, UF=false, thetaF=false, VF=false,
            force_ONB=false)

Compute the one-period unitary and its quasienergy spectrum. `evolution`
accepts `H,T`; `H,t_list,dt_list`; or `H_list,dt_list` entries using either
symbol or string dictionary keys.
"""
struct Floquet
    T::Float64
    EF::Vector{Float64}
    HF::Union{Nothing,Matrix{ComplexF64}}
    UF::Union{Nothing,Matrix{ComplexF64}}
    VF::Union{Nothing,Matrix{ComplexF64}}
    thetaF::Union{Nothing,Vector{ComplexF64}}
end

function _dict_get(dictionary::AbstractDict, key::Symbol, default=nothing)
    haskey(dictionary, key) && return dictionary[key]
    string_key = String(key)
    return haskey(dictionary, string_key) ? dictionary[string_key] : default
end
_dict_has(dictionary::AbstractDict, key::Symbol) =
    haskey(dictionary, key) || haskey(dictionary, String(key))
_floquet_operator(H::Hamiltonian, time=0.0) = tocsc(H; time)
_floquet_operator(H::AbstractMatrix, time=0.0) = H
_floquet_operator(H, time=0.0) =
    applicable(H, time) ? _floquet_operator(H(time), time) : H

function _parallel_for(f, count::Int, n_jobs::Integer)
    workers = min(max(1, Int(n_jobs)), Threads.nthreads(), count)
    if workers <= 1
        for index in 1:count
            f(index)
        end
        return
    end
    tasks = [
        Threads.@spawn begin
            for index in worker:workers:count
                f(index)
            end
        end
        for worker in 1:workers
    ]
    foreach(fetch, tasks)
    return
end

function _prefer_dense_floquet(operator::AbstractMatrix)
    dimension = size(operator, 1)
    dimension <= 64 && return true
    operator isa StridedMatrix && return true
    density = nnz(sparse(operator)) / max(1, dimension^2)
    return density >= 0.15
end

function _apply_floquet_step(
    operator::AbstractMatrix,
    unitary::AbstractMatrix,
    duration;
    n_jobs::Integer=1,
)
    size(operator, 1) == size(operator, 2) == size(unitary, 1) ||
        throw(DimensionMismatch("Floquet step dimensions do not match"))
    scale = -im * duration
    if _prefer_dense_floquet(operator)
        return exp(scale .* Matrix(operator)) * unitary
    end
    T = promote_type(ComplexF64, eltype(operator), eltype(unitary))
    result = Matrix{T}(undef, size(unitary))
    _parallel_for(size(unitary, 2), n_jobs) do column
        state = _krylov_expmv(
            operator,
            @view(unitary[:, column]),
            convert(T, scale),
        )
        copyto!(@view(result[:, column]), state)
    end
    return result
end

function _step_unitary(matrices, durations; n_jobs::Integer=1)
    if applicable(length, matrices) && applicable(length, durations)
        length(matrices) == length(durations) ||
            throw(DimensionMismatch(
                "Hamiltonian and duration lists must have equal lengths",
            ))
    end
    matrix_iteration = iterate(matrices)
    duration_iteration = iterate(durations)
    (matrix_iteration === nothing) == (duration_iteration === nothing) ||
        throw(DimensionMismatch(
            "Hamiltonian and duration lists must have equal lengths",
        ))
    matrix_iteration === nothing &&
        throw(ArgumentError("at least one evolution step is required"))
    first_matrix, matrix_state = matrix_iteration
    first_duration, duration_state = duration_iteration
    first_operator = _floquet_operator(first_matrix)
    size(first_operator, 1) == size(first_operator, 2) ||
        throw(ArgumentError("Hamiltonians must be square"))
    unitary = Matrix{ComplexF64}(I, size(first_operator)...)
    unitary = _apply_floquet_step(
        first_operator,
        unitary,
        first_duration;
        n_jobs,
    )
    while true
        matrix_iteration = iterate(matrices, matrix_state)
        duration_iteration = iterate(durations, duration_state)
        (matrix_iteration === nothing) == (duration_iteration === nothing) ||
            throw(DimensionMismatch(
                "Hamiltonian and duration lists must have equal lengths",
            ))
        matrix_iteration === nothing && break
        matrix, matrix_state = matrix_iteration
        duration, duration_state = duration_iteration
        operator = _floquet_operator(matrix)
        size(operator) == size(first_operator) ||
            throw(DimensionMismatch("all Hamiltonians must have the same shape"))
        unitary = _apply_floquet_step(
            operator,
            unitary,
            duration;
            n_jobs,
        )
    end
    return unitary isa Matrix{ComplexF64} ?
        unitary :
        Matrix{ComplexF64}(unitary)
end

function _continuous_floquet_unitary(
    H,
    period::Real;
    n_jobs::Integer=1,
    max_step::Real=0.01,
)
    first_matrix = _floquet_operator(H, 0.0)
    dimension = size(first_matrix, 1)
    size(first_matrix) == (dimension, dimension) ||
        throw(ArgumentError("Hamiltonian must be square"))
    unitary = Matrix{ComplexF64}(undef, dimension, dimension)
    _parallel_for(dimension, n_jobs) do column
        initial = zeros(ComplexF64, dimension)
        initial[column] = 1
        evolved = if H isa Hamiltonian
            evolve(H, initial, 0.0, [period]; max_step)
        else
            derivative = (time, state) ->
                -im .* (_floquet_operator(H, time) * state)
            evolve(
                initial,
                0.0,
                [period],
                derivative;
                max_step,
            )
        end
        copyto!(@view(unitary[:, column]), @view(evolved[:, end]))
    end
    return unitary
end

function Floquet(
    evolution::AbstractDict;
    HF::Bool=false,
    UF::Bool=false,
    thetaF::Bool=false,
    VF::Bool=false,
    n_jobs::Integer=1,
    force_ONB::Bool=false,
)
    n_jobs > 0 || throw(ArgumentError("n_jobs must be positive"))
    period, unitary = if _dict_has(evolution, :H_list) &&
                         _dict_has(evolution, :dt_list)
        H_list = collect(_dict_get(evolution, :H_list))
        dt_list = collect(_dict_get(evolution, :dt_list))
        length(H_list) == length(dt_list) ||
            throw(DimensionMismatch("H_list and dt_list must have equal lengths"))
        T = float(_dict_get(evolution, :T, sum(dt_list)))
        T > 0 || throw(ArgumentError("the Floquet period must be positive"))
        T, _step_unitary(H_list, dt_list; n_jobs)
    elseif _dict_has(evolution, :H) &&
           _dict_has(evolution, :t_list) &&
           _dict_has(evolution, :dt_list)
        H = _dict_get(evolution, :H)
        t_list = collect(_dict_get(evolution, :t_list))
        dt_list = collect(_dict_get(evolution, :dt_list))
        length(t_list) == length(dt_list) ||
            throw(DimensionMismatch("t_list and dt_list must have equal lengths"))
        T = float(_dict_get(evolution, :T, sum(dt_list)))
        T > 0 || throw(ArgumentError("the Floquet period must be positive"))
        matrices = (_floquet_operator(H, time) for time in t_list)
        T, _step_unitary(matrices, dt_list; n_jobs)
    elseif _dict_has(evolution, :H) && _dict_has(evolution, :T)
        requested_H = _dict_get(evolution, :H)
        T = float(_dict_get(evolution, :T))
        T > 0 || throw(ArgumentError("the Floquet period must be positive"))
        is_dynamic =
            applicable(requested_H, 0.0) ||
            (requested_H isa Hamiltonian &&
             !isempty(requested_H.dynamic_terms))
        if is_dynamic
            max_step = float(_dict_get(evolution, :max_step, 0.01))
            T, _continuous_floquet_unitary(
                requested_H,
                T;
                n_jobs,
                max_step,
            )
        else
            T, _step_unitary((requested_H,), (T,); n_jobs)
        end
    else
        throw(ArgumentError("unsupported Floquet evolution dictionary"))
    end
    period > 0 || throw(ArgumentError("the Floquet period must be positive"))

    vectors = nothing
    phases = if VF || force_ONB || HF
        decomposition = eigen(unitary)
        vectors = ComplexF64.(decomposition.vectors)
        ComplexF64.(decomposition.values)
    else
        ComplexF64.(eigvals(unitary))
    end
    energies = real.((im / period) .* log.(phases))
    order = sortperm(energies)
    energies = Float64.(energies[order])
    phases = phases[order]
    vectors === nothing || (vectors = vectors[:, order])
    effective = if HF
        spectral_effective =
            vectors * Diagonal(ComplexF64.(energies)) * vectors'
        ComplexF64.((spectral_effective + spectral_effective') ./ 2)
    else
        nothing
    end
    if force_ONB && vectors !== nothing
        vectors = Matrix(qr(vectors).Q)
    end

    return Floquet(
        period,
        energies,
        effective,
        UF ? unitary : nothing,
        VF ? vectors : nothing,
        thetaF ? phases : nothing,
    )
end

function _normalize_block_terms(static)
    all(term -> term isa OperatorTerm, static) &&
        return OperatorTerm[static...]
    return OperatorTerm[
        OperatorTerm(String(term[1]), collect(term[2]))
        for term in static
    ]
end

struct _BlockDiagonalBasis <: AbstractBasis
    dimension::Int
end

Base.length(basis::_BlockDiagonalBasis) = basis.dimension

function _normalize_block_dynamic(basis, dynamic, dtype::Type)
    isempty(dynamic) && return Any[]
    reduced_dimension = length(basis)
    projector = nothing
    normalized = Any[]
    for entry in dynamic
        (entry isa Tuple || entry isa AbstractVector) ||
            throw(ArgumentError("dynamic entries must be tuples or vectors"))
        if first(entry) isa AbstractMatrix
            length(entry) == 3 ||
                throw(ArgumentError(
                    "dynamic matrix entries are [matrix, f, f_args]",
                ))
            matrix, function_value, arguments = entry
            reduced_matrix = if size(matrix) ==
                                (reduced_dimension, reduced_dimension)
                matrix
            else
                projector === nothing &&
                    (projector = projection_matrix(
                        basis,
                        dtype;
                        sparse=true,
                    ))
                size(matrix) == (size(projector, 1), size(projector, 1)) ||
                    throw(DimensionMismatch(
                        "dynamic matrix must act in the block or parent basis",
                    ))
                projector' * matrix * projector
            end
            push!(normalized, Any[reduced_matrix, function_value, arguments])
        else
            length(entry) == 4 ||
                throw(ArgumentError(
                    "dynamic operator entries are [op, couplings, f, f_args]",
                ))
            push!(normalized, entry)
        end
    end
    return normalized
end

function _construct_block_hamiltonian(
    basis,
    static,
    dynamic,
    dtype::Type;
    check_symm::Bool,
    check_herm::Bool,
    check_pcon::Bool,
)
    normalized_dynamic = _normalize_block_dynamic(basis, dynamic, dtype)
    return Hamiltonian(
        static,
        normalized_dynamic;
        basis,
        dtype,
        static_fmt=:csc,
        dynamic_fmt=:csc,
        check_symm,
        check_herm,
        check_pcon,
    )
end

function _block_keywords(values)
    keywords = Dict{Symbol,Any}()
    for (key, value) in values
        name = Symbol(key)
        name === :Nup && (name = :nup)
        keywords[name] = value
    end
    return keywords
end

function _construct_block_basis(basis_con, basis_args, basis_kwargs, block)
    block isa AbstractBasis && return block
    block isa AbstractDict ||
        throw(ArgumentError("each block must be a basis or dictionary"))
    keywords = merge(_block_keywords(basis_kwargs), _block_keywords(block))
    return basis_con(basis_args...; keywords...)
end

function _sparse_block_diagonal(matrices, ::Type{T}) where {T<:Number}
    sparse_matrices = SparseMatrixCSC{T,Int}[
        SparseMatrixCSC{T,Int}(sparse(matrix))
        for matrix in matrices
    ]
    isempty(sparse_matrices) &&
        throw(ArgumentError("at least one block is required"))
    return blockdiag(sparse_matrices...)
end

function _block_projector_sparse(get_proj_kwargs)
    keywords = _block_keywords(get_proj_kwargs)
    unsupported = setdiff(keys(keywords), (:sparse,))
    isempty(unsupported) ||
        throw(ArgumentError("unsupported projector keyword(s): $unsupported"))
    return Bool(get(keywords, :sparse, true))
end

"""
    block_diag_hamiltonian(blocks, static, dynamic, basis_con, basis_args,
                           dtype; get_proj=true, ...)

Build symmetry-sector bases, their projectors, and the corresponding sparse
block-diagonal Hamiltonian.
"""
function block_diag_hamiltonian(
    blocks,
    static,
    dynamic,
    basis_con,
    basis_args,
    dtype::Type;
    basis_kwargs::AbstractDict=Dict(),
    get_proj_kwargs::AbstractDict=Dict(),
    get_proj::Bool=true,
    check_symm::Bool=true,
    check_herm::Bool=true,
    check_pcon::Bool=true,
)
    terms = _normalize_block_terms(static)
    bases = [
        _construct_block_basis(basis_con, basis_args, basis_kwargs, block)
        for block in blocks
    ]
    hamiltonians = [
        _construct_block_hamiltonian(
            basis,
            terms,
            dynamic,
            dtype;
            check_symm,
            check_herm,
            check_pcon,
        )
        for basis in bases
    ]
    static_matrix = _sparse_block_diagonal(
        (hamiltonian.data for hamiltonian in hamiltonians),
        dtype,
    )
    result = if isempty(dynamic)
        static_matrix
    else
        dynamic_entries = Any[]
        dynamic_count = length(first(hamiltonians).dynamic_terms)
        all(
            length(hamiltonian.dynamic_terms) == dynamic_count
            for hamiltonian in hamiltonians
        ) || throw(ArgumentError(
            "each symmetry block must contain the same dynamic terms",
        ))
        for index in 1:dynamic_count
            matrices = [
                hamiltonian.dynamic_terms[index][1]
                for hamiltonian in hamiltonians
            ]
            function_value = hamiltonians[1].dynamic_terms[index][2]
            arguments = hamiltonians[1].dynamic_terms[index][3]
            all(
                hamiltonian.dynamic_terms[index][2] === function_value &&
                hamiltonian.dynamic_terms[index][3] == arguments
                for hamiltonian in hamiltonians
            ) || throw(ArgumentError(
                "dynamic terms must have a consistent order across blocks",
            ))
            push!(
                dynamic_entries,
                Any[
                    _sparse_block_diagonal(matrices, dtype),
                    function_value,
                    arguments,
                ],
            )
        end
        Hamiltonian(
            Any[static_matrix],
            dynamic_entries;
            basis=_BlockDiagonalBasis(size(static_matrix, 1)),
            dtype,
            static_fmt=:csc,
            dynamic_fmt=:csc,
            check_symm=false,
            check_herm=false,
            check_pcon=false,
        )
    end
    get_proj || return result
    sparse_projectors = _block_projector_sparse(get_proj_kwargs)
    projectors = [
        projection_matrix(basis, dtype; sparse=sparse_projectors)
        for basis in bases
    ]
    return hcat(projectors...), result
end

mutable struct BlockOps
    basis_dict::Dict{String,Any}
    H_dict::Dict{String,Any}
    P_dict::Dict{String,Any}
    dtype::DataType
    static::Vector{OperatorTerm}
    dynamic::Vector{Any}
    save_previous_data::Bool
    sparse_projectors::Bool
    basis_kwargs::Dict{Any,Any}
    check_symm::Bool
    check_herm::Bool
    check_pcon::Bool
end

function BlockOps(
    blocks,
    static,
    dynamic,
    basis_con,
    basis_args,
    dtype::Type;
    basis_kwargs::AbstractDict=Dict(),
    get_proj_kwargs::AbstractDict=Dict(),
    save_previous_data::Bool=true,
    compute_all_blocks::Bool=false,
    check_symm::Bool=true,
    check_herm::Bool=true,
    check_pcon::Bool=true,
)
    bases = Dict{String,Any}()
    for block in blocks
        basis = _construct_block_basis(basis_con, basis_args, basis_kwargs, block)
        length(basis) > 0 && (bases[repr(block)] = basis)
    end
    operator = BlockOps(
        bases,
        Dict{String,Any}(),
        Dict{String,Any}(),
        dtype,
        _normalize_block_terms(static),
        Any[dynamic...],
        save_previous_data || compute_all_blocks,
        _block_projector_sparse(get_proj_kwargs),
        Dict{Any,Any}(basis_kwargs),
        check_symm,
        check_herm,
        check_pcon,
    )
    compute_all_blocks && compute_all_blocks!(operator)
    return operator
end

function compute_all_blocks!(operator::BlockOps)
    for (key, basis) in operator.basis_dict
        haskey(operator.P_dict, key) ||
            (operator.P_dict[key] =
                projection_matrix(
                    basis,
                    operator.dtype;
                    sparse=operator.sparse_projectors,
                ))
        haskey(operator.H_dict, key) ||
            (operator.H_dict[key] = _construct_block_hamiltonian(
                basis,
                operator.static,
                operator.dynamic,
                operator.dtype;
                check_symm=operator.check_symm,
                check_herm=operator.check_herm,
                check_pcon=operator.check_pcon,
            ))
    end
    return operator
end

function update_blocks!(
    operator::BlockOps,
    blocks,
    basis_con,
    basis_args;
    compute_all_blocks::Bool=false,
)
    for block in blocks
        key = repr(block)
        haskey(operator.basis_dict, key) && continue
        basis = _construct_block_basis(
            basis_con,
            basis_args,
            operator.basis_kwargs,
            block,
        )
        length(basis) > 0 && (operator.basis_dict[key] = basis)
    end
    compute_all_blocks && compute_all_blocks!(operator)
    return operator
end

function _block_data(operator::BlockOps, key, basis)
    if !operator.save_previous_data
        return (
            projection_matrix(
                basis,
                operator.dtype;
                sparse=operator.sparse_projectors,
            ),
            _construct_block_hamiltonian(
                basis,
                operator.static,
                operator.dynamic,
                operator.dtype;
                check_symm=operator.check_symm,
                check_herm=operator.check_herm,
                check_pcon=operator.check_pcon,
            ),
        )
    end
    projector = get!(
        () -> projection_matrix(
            basis,
            operator.dtype;
            sparse=operator.sparse_projectors,
        ),
        operator.P_dict,
        key,
    )
    hamiltonian = get!(
        () -> _construct_block_hamiltonian(
            basis,
            operator.static,
            operator.dynamic,
            operator.dtype;
            check_symm=operator.check_symm,
            check_herm=operator.check_herm,
            check_pcon=operator.check_pcon,
        ),
        operator.H_dict,
        key,
    )
    return projector, hamiltonian
end

function _active_block_entries(
    operator::BlockOps,
    psi_0::AbstractVector;
    H_time_eval::Real=0.0,
    shift=nothing,
    freeze_dynamic::Bool=false,
)
    entries = Any[]
    for (key, basis) in operator.basis_dict
        projector, hamiltonian = _block_data(operator, key, basis)
        projected = projector' * psi_0
        norm(projected) > 1000eps(Float64) || continue
        generator = if shift === nothing && !freeze_dynamic
            hamiltonian
        else
            matrix = tocsc(hamiltonian; time=H_time_eval)
            shift === nothing ?
                matrix :
                matrix + spdiagm(
                    0 => fill(
                        convert(eltype(matrix), shift),
                        size(matrix, 1),
                    ),
                )
        end
        push!(entries, (projector, generator, projected))
    end
    isempty(entries) &&
        throw(ArgumentError(
            "initial state has no projection onto the selected blocks",
        ))
    return entries
end

function _combine_block_entries(entries)
    projectors = [entry[1] for entry in entries]
    matrices = [
        entry[2] isa Hamiltonian ?
        tocsc(entry[2]) :
        SparseMatrixCSC(sparse(entry[2]))
        for entry in entries
    ]
    initial = reduce(vcat, (entry[3] for entry in entries))
    return Any[(hcat(projectors...), blockdiag(matrices...), initial)]
end

function _accumulate_projection!(output, projector, states)
    if applicable(mul!, output, projector, states, true, true)
        mul!(output, projector, states, true, true)
    else
        output .+= projector * states
    end
    return output
end

function _block_batch_states(
    entries,
    scales,
    n_jobs::Integer;
    tol::Real=1e-12,
    krylov_dim::Integer=30,
)
    states = Vector{Any}(undef, length(entries))
    _parallel_for(length(entries), n_jobs) do index
        entry = entries[index]
        states[index] = _krylov_expmv_times(
            entry[2],
            entry[3],
            scales;
            tol,
            krylov_dim,
        )
    end
    return states
end

struct _BlockEvolutionIterator{D,S,T<:Number}
    entries::D
    scales::S
    dimension::Int
    n_jobs::Int
    tol::Float64
    krylov_dim::Int
    output_type::Type{T}
end

Base.IteratorSize(::Type{<:_BlockEvolutionIterator}) = Base.HasLength()
Base.length(iterator::_BlockEvolutionIterator) = length(iterator.scales)
Base.eltype(::Type{_BlockEvolutionIterator{D,S,T}}) where {D,S,T} =
    Vector{T}

function Base.iterate(iterator::_BlockEvolutionIterator, index::Int=1)
    index > length(iterator) && return nothing
    states = Vector{Any}(undef, length(iterator.entries))
    scale = iterator.scales[index]
    _parallel_for(length(iterator.entries), iterator.n_jobs) do block
        entry = iterator.entries[block]
        states[block] = _krylov_expmv(
            entry[2],
            entry[3],
            scale;
            tol=iterator.tol,
            krylov_dim=iterator.krylov_dim,
        )
    end
    result = zeros(iterator.output_type, iterator.dimension)
    for (entry, state) in zip(iterator.entries, states)
        _accumulate_projection!(result, entry[1], state)
    end
    return result, index + 1
end

function _block_evolution(
    entries,
    scales,
    dimension::Int,
    output_type::Type{T};
    iterate::Bool=false,
    n_jobs::Integer=1,
    block_diag::Bool=false,
    tol::Real=1e-12,
    krylov_dim::Integer=30,
) where {T<:Number}
    selected_entries = block_diag ? _combine_block_entries(entries) : entries
    if iterate
        return _BlockEvolutionIterator(
            selected_entries,
            scales,
            dimension,
            Int(n_jobs),
            Float64(tol),
            Int(krylov_dim),
            output_type,
        )
    end
    result = zeros(output_type, dimension, length(scales))
    states = _block_batch_states(
        selected_entries,
        scales,
        n_jobs;
        tol,
        krylov_dim,
    )
    for (entry, block_states) in zip(selected_entries, states)
        _accumulate_projection!(result, entry[1], block_states)
    end
    return result
end

function _dynamic_block_evolution(
    entries,
    t0::Real,
    targets,
    dimension::Int,
    output_type::Type{T};
    iterate::Bool=false,
    n_jobs::Integer=1,
    imag_time::Bool=false,
    kwargs...,
) where {T<:Number}
    block_states = Vector{Any}(undef, length(entries))
    _parallel_for(length(entries), n_jobs) do index
        projector, hamiltonian, initial = entries[index]
        block_states[index] = evolve(
            hamiltonian,
            initial,
            t0,
            targets;
            iterate=false,
            imag_time,
            kwargs...,
        )
    end
    result = zeros(output_type, dimension, length(targets))
    for (entry, states) in zip(entries, block_states)
        _accumulate_projection!(result, entry[1], states)
    end
    iterate &&
        return (copy(@view(result[:, index])) for index in axes(result, 2))
    return result
end

function evolve(
    operator::BlockOps,
    psi_0::AbstractVector,
    t0::Real,
    times;
    iterate::Bool=false,
    n_jobs::Integer=1,
    block_diag::Bool=false,
    stack_state::Bool=false,
    imag_time::Bool=false,
    kwargs...,
)
    n_jobs > 0 || throw(ArgumentError("n_jobs must be positive"))
    scalar_target = times isa Number
    targets = scalar_target ? [times] : collect(times)
    entries = _active_block_entries(operator, psi_0)
    output_type = promote_type(ComplexF64, eltype(psi_0))
    if !isempty(operator.dynamic)
        output = _dynamic_block_evolution(
            entries,
            t0,
            targets,
            length(psi_0),
            output_type;
            iterate,
            n_jobs,
            imag_time,
            kwargs...,
        )
        iterate && return output
        return scalar_target ? copy(@view(output[:, 1])) : output
    end
    unsupported = setdiff(keys(kwargs), (:tol, :krylov_dim))
    isempty(unsupported) ||
        throw(ArgumentError("unsupported block evolution keyword(s): $unsupported"))
    scales = imag_time ?
        ComplexF64[-(time - t0) for time in targets] :
        ComplexF64[-im * (time - t0) for time in targets]
    output = _block_evolution(
        entries,
        scales,
        length(psi_0),
        output_type;
        iterate,
        n_jobs,
        block_diag,
        tol=get(kwargs, :tol, 1e-12),
        krylov_dim=get(kwargs, :krylov_dim, 30),
    )
    iterate && return output
    return scalar_target ? copy(@view(output[:, 1])) : output
end

function block_expm(
    operator::BlockOps,
    psi_0::AbstractVector;
    H_time_eval::Real=0.0,
    iterate::Bool=false,
    n_jobs::Integer=1,
    block_diag::Bool=false,
    a::Number=-im,
    start=nothing,
    stop=nothing,
    endpoint::Bool=true,
    num::Integer=50,
    shift=nothing,
)
    n_jobs > 0 || throw(ArgumentError("n_jobs must be positive"))
    scales = start === nothing && stop === nothing ?
        [one(float(real(a)))] :
        collect(range(start, stop; length=num + (!endpoint)))
    !endpoint && pop!(scales)
    entries = _active_block_entries(
        operator,
        psi_0;
        H_time_eval,
        shift,
        freeze_dynamic=true,
    )
    output_type = promote_type(ComplexF64, eltype(psi_0), typeof(a))
    output = _block_evolution(
        entries,
        scales .* a,
        length(psi_0),
        output_type;
        iterate,
        n_jobs,
        block_diag,
    )
    iterate && return output
    return length(scales) == 1 ? vec(output) : output
end

function _lanczos_action(A, vector)
    result = A * vector
    result isa AbstractVector ||
        throw(ArgumentError("A must act on a vector and return a vector"))
    return result
end

function _lanczos_action!(output, A, vector)
    if applicable(mul!, output, A, vector)
        mul!(output, A, vector)
    else
        copyto!(output, _lanczos_action(A, vector))
    end
    return output
end

"""
    lanczos_full(A, v0, m; full_ortho=false, out=nothing, eps=nothing)

Construct a Hermitian Krylov basis and return `(E, V, Q_T)`, where `Q_T`
stores Lanczos vectors in rows.
"""
function lanczos_full(
    A,
    v0::AbstractVector,
    m::Integer;
    full_ortho::Bool=false,
    out=nothing,
    eps=nothing,
)
    n = length(v0)
    1 <= m < n ||
        throw(ArgumentError("Lanczos dimension m must satisfy 1 <= m < length(v0)"))
    initial_norm = norm(v0)
    initial_norm > 0 ||
        throw(ArgumentError("initial Lanczos vector must be nonzero"))

    first_action = _lanczos_action(A, v0)
    T = promote_type(eltype(v0), eltype(first_action))
    RT = typeof(real(zero(T)))
    tolerance = eps === nothing ? Base.eps(RT) : convert(RT, eps)
    Q_T = out === nothing ? Matrix{T}(undef, m, n) : out
    size(Q_T) == (m, n) ||
        throw(DimensionMismatch("out must have shape (m, length(v0))"))

    q = Vector{T}(v0)
    q ./= initial_norm
    q_previous = zeros(T, n)
    residual = Vector{T}(first_action)
    residual ./= initial_norm
    beta_previous = zero(RT)
    alphas = Vector{RT}(undef, m)
    betas = Vector{RT}(undef, m - 1)
    used = 0

    for index in 1:m
        copyto!(@view(Q_T[index, :]), q)
        used = index
        index > 1 && _lanczos_action!(residual, A, q)
        index > 1 && (residual .-= beta_previous .* q_previous)
        alpha = real(dot(q, residual))
        residual .-= alpha .* q
        if full_ortho
            for previous in 1:index
                q_basis = @view Q_T[previous, :]
                residual .-= dot(q_basis, residual) .* q_basis
            end
        end
        alphas[index] = alpha
        index == m && break
        beta = norm(residual)
        beta <= tolerance && break
        betas[index] = beta
        copyto!(q_previous, q)
        @. q = residual / beta
        beta_previous = beta
    end

    tridiagonal = SymTridiagonal(
        alphas[1:used],
        betas[1:max(0, used - 1)],
    )
    decomposition = eigen(tridiagonal)
    vectors = used == m ? Q_T : @view(Q_T[1:used, :])
    return decomposition.values, decomposition.vectors, vectors
end

function _lanczos_coefficients(A, v0::AbstractVector, m::Integer, eps)
    n = length(v0)
    1 <= m < n ||
        throw(ArgumentError("Lanczos dimension m must satisfy 1 <= m < length(v0)"))
    initial_norm = norm(v0)
    initial_norm > 0 ||
        throw(ArgumentError("initial Lanczos vector must be nonzero"))
    first_action = _lanczos_action(A, v0)
    T = promote_type(eltype(v0), eltype(first_action))
    RT = typeof(real(zero(T)))
    tolerance = eps === nothing ? Base.eps(RT) : convert(RT, eps)
    q = Vector{T}(v0)
    q ./= initial_norm
    q_previous = zeros(T, n)
    residual = Vector{T}(first_action)
    residual ./= initial_norm
    alphas = Vector{RT}(undef, m)
    betas = Vector{RT}(undef, m - 1)
    used = 0
    beta_previous = zero(RT)
    for index in 1:m
        used = index
        index > 1 && _lanczos_action!(residual, A, q)
        index > 1 && (residual .-= beta_previous .* q_previous)
        alpha = real(dot(q, residual))
        residual .-= alpha .* q
        alphas[index] = alpha
        index == m && break
        beta = norm(residual)
        beta <= tolerance && break
        betas[index] = beta
        copyto!(q_previous, q)
        @. q = residual / beta
        beta_previous = beta
    end
    return alphas[1:used], betas[1:max(0, used - 1)], T
end

struct _LanczosVectorIterator{A,V,CA,CB,T}
    operator::A
    initial::V
    alphas::CA
    betas::CB
    vector_type::Type{T}
end

Base.IteratorSize(::Type{<:_LanczosVectorIterator}) = Base.HasLength()
Base.length(iterator::_LanczosVectorIterator) = length(iterator.alphas)
Base.eltype(::Type{_LanczosVectorIterator{A,V,CA,CB,T}}) where {A,V,CA,CB,T} =
    Vector{T}

function Base.iterate(iterator::_LanczosVectorIterator)
    q = Vector{iterator.vector_type}(iterator.initial)
    q ./= norm(q)
    q_previous = zeros(iterator.vector_type, length(q))
    residual = similar(q)
    return copy(q), (1, q_previous, q, residual)
end

function Base.iterate(iterator::_LanczosVectorIterator, state)
    index, q_previous, q, residual = state
    index >= length(iterator) && return nothing
    _lanczos_action!(residual, iterator.operator, q)
    index > 1 &&
        (residual .-= iterator.betas[index - 1] .* q_previous)
    residual .-= iterator.alphas[index] .* q
    q_next = q_previous
    @. q_next = residual / iterator.betas[index]
    return copy(q_next), (index + 1, q, q_next, residual)
end

"""
    lanczos_iter(A, v0, m; return_vec_iter=true, ...)

Memory-oriented interface compatible with QuSpin. The Julia iterator yields
immutable copies of the rows produced by `lanczos_full`.
"""
function lanczos_iter(
    A,
    v0::AbstractVector,
    m::Integer;
    return_vec_iter::Bool=true,
    copy_v0::Bool=true,
    copy_A::Bool=false,
    eps=nothing,
)
    alphas, betas, T = _lanczos_coefficients(A, v0, m, eps)
    decomposition = eigen(SymTridiagonal(alphas, betas))
    return_vec_iter || return decomposition.values, decomposition.vectors
    initial = copy_v0 ? copy(v0) : v0
    operator = copy_A ? deepcopy(A) : A
    iterator = _LanczosVectorIterator(
        operator,
        initial,
        alphas,
        betas,
        T,
    )
    return decomposition.values, decomposition.vectors, iterator
end

"""Compute a coefficient-weighted linear combination of Lanczos rows."""
function lin_comb_Q_T(coeff, Q_T; out=nothing)
    if Q_T isa AbstractMatrix
        length(coeff) == size(Q_T, 1) ||
            throw(DimensionMismatch("one coefficient is required per Lanczos vector"))
        T = promote_type(eltype(coeff), eltype(Q_T))
        result = out === nothing ? Vector{T}(undef, size(Q_T, 2)) : out
        axes(result) == (axes(Q_T, 2),) ||
            throw(DimensionMismatch("out must match a Lanczos vector"))
        mul!(result, transpose(Q_T), coeff)
        return result
    end
    iteration = iterate(Q_T)
    iteration === nothing &&
        throw(ArgumentError("Lanczos vector iterator must be nonempty"))
    first_vector, iterator_state = iteration
    length(coeff) > 0 ||
        throw(DimensionMismatch("one coefficient is required per Lanczos vector"))
    T = promote_type(eltype(coeff), eltype(first_vector))
    result = out === nothing ? similar(first_vector, T) : out
    axes(result) == axes(first_vector) ||
        throw(DimensionMismatch("out must match a Lanczos vector"))
    @. result = coeff[1] * first_vector
    index = 1
    while true
        iteration = iterate(Q_T, iterator_state)
        iteration === nothing && break
        vector, iterator_state = iteration
        index += 1
        index <= length(coeff) ||
            throw(DimensionMismatch("too many Lanczos vectors"))
        axpy!(coeff[index], vector, result)
    end
    index == length(coeff) ||
        throw(DimensionMismatch("too few Lanczos vectors"))
    return result
end

"""Apply a matrix exponential to the Lanczos starting vector."""
function expm_lanczos(E, V, Q_T; a::Number=1, out=nothing)
    length(E) == size(V, 1) == size(V, 2) ||
        throw(DimensionMismatch("E and V dimensions must agree"))
    coefficients = V * (exp.(a .* E) .* V[1, :])
    return lin_comb_Q_T(coefficients, Q_T; out)
end

function _lanczos_rows(Q_T)
    return Q_T isa AbstractMatrix ? eachrow(Q_T) : Q_T
end

_temperature_output(values, beta) = beta isa Number ? only(values) : values

"""
    ftlm_static_iteration(observables, E, V, Q_T; beta=0)

One finite-temperature Lanczos trace-estimation iteration. Returns a dictionary
of unnormalized observable estimates and the corresponding identity estimate.
"""
function ftlm_static_iteration(observables::AbstractDict, E, V, Q_T; beta=0)
    m = length(E)
    size(V) == (m, m) ||
        throw(DimensionMismatch("Lanczos inputs must share the same Krylov dimension"))
    betas = beta isa Number ? [beta] : collect(beta)
    probabilities = exp.(-E .* transpose(betas))
    coefficients = V * (V[1, :] .* probabilities)
    vectors = _lanczos_rows(Q_T)
    iteration = iterate(vectors)
    iteration === nothing &&
        throw(ArgumentError("Lanczos vector iterator must be nonempty"))
    initial, iterator_state = iteration
    applied = Dict(
        key => observable * initial
        for (key, observable) in observables
    )
    results = Dict{Any,Any}(
        key => zeros(
            promote_type(eltype(coefficients), eltype(value)),
            length(betas),
        )
        for (key, value) in applied
    )
    index = 1
    while true
        for key in keys(applied)
            overlap = dot(initial, applied[key])
            results[key] .+= overlap .* @view(coefficients[index, :])
        end
        iteration = iterate(vectors, iterator_state)
        iteration === nothing && break
        initial, iterator_state = iteration
        index += 1
        index <= m ||
            throw(DimensionMismatch("too many Lanczos vectors"))
    end
    index == m || throw(DimensionMismatch("too few Lanczos vectors"))
    for (key, observable) in observables
        results[key] = _temperature_output(results[key], beta)
    end
    identity = _temperature_output(vec(coefficients[1, :]), beta)
    return results, identity
end

"""
    ltlm_static_iteration(observables, E, V, Q_T; beta=0)

One symmetric low-temperature Lanczos trace-estimation iteration.
"""
function ltlm_static_iteration(observables::AbstractDict, E, V, Q_T; beta=0)
    m = length(E)
    size(V) == (m, m) ||
        throw(DimensionMismatch("Lanczos inputs must share the same Krylov dimension"))
    betas = beta isa Number ? [beta] : collect(beta)
    isempty(betas) && throw(ArgumentError("beta must be nonempty"))
    minimum_beta = minimum(betas)
    nv = m
    if minimum_beta > 0
        shifted = E .- minimum(E)
        half_boltzmann = exp.(-shifted .* minimum_beta ./ 2)
        cutoff = findlast(>(eps(eltype(half_boltzmann))), half_boltzmann)
        nv = cutoff === nothing ? 1 : cutoff
    end
    energies = @view E[1:nv]
    eigenvectors = @view V[:, 1:nv]
    weights =
        eigenvectors[1, :] .*
        exp.(-energies .* transpose(betas) ./ 2)
    results = Dict{Any,Any}()

    first_ritz = lin_comb_Q_T(@view(eigenvectors[:, 1]), _lanczos_rows(Q_T))
    ritz_vectors = Matrix{eltype(first_ritz)}(
        undef,
        length(first_ritz),
        nv,
    )
    copyto!(@view(ritz_vectors[:, 1]), first_ritz)
    for column in 2:nv
        lin_comb_Q_T(
            @view(eigenvectors[:, column]),
            _lanczos_rows(Q_T);
            out=@view(ritz_vectors[:, column]),
        )
    end
    for (key, observable) in observables
        projected = ritz_vectors' * (observable * ritz_vectors)
        values = [
            dot(@view(weights[:, index]), projected * @view(weights[:, index]))
            for index in axes(weights, 2)
        ]
        results[key] = _temperature_output(values, beta)
    end

    identity_values = [
        sum(
            abs2.(eigenvectors[1, :]) .*
            exp.(-betas[index] .* energies),
        )
        for index in eachindex(betas)
    ]
    return results, _temperature_output(identity_values, beta)
end

end
