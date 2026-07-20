module Tools

using LinearAlgebra
using ..Basis: AbstractBasis, FixedUInt, get_basis_type, projection_matrix
using ..Operators: Hamiltonian, OperatorTerm, _krylov_expmv, toarray
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
    isapprox(sum(p1), one(sum(p1)); atol=1e-13, rtol=0) ||
        throw(ArgumentError("p1 must be normalized"))
    isapprox(sum(p2), one(sum(p2)); atol=1e-13, rtol=0) ||
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
    if !allunique(energies)
        verbose && @warn "degeneracies found in energy spectrum"
        return NaN
    end
    spacings = diff(energies)
    return sum(
        min(spacings[index], spacings[index + 1]) /
        max(spacings[index], spacings[index + 1])
        for index in 1:(length(spacings) - 1)
    ) / (length(spacings) - 1)
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
    for (row, value) in pairs(values)
        encoded = BigInt(value)
        encoded >= 0 || throw(ArgumentError("basis integers must be nonnegative"))
        for column in 1:width
            result[row, column] = UInt8((encoded >> (width - column)) & 1)
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
    for row in axes(states, 1)
        value = BigInt(0)
        for bit in @view states[row, :]
            value = (value << 1) | Int(bit)
        end
        result[row] = T(value)
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
    result = a * (array * other)
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

function _projector(proj::AbstractMatrix, ::Type{T}) where {T<:Number}
    return Matrix{T}(proj)
end
_projector(proj::AbstractBasis, ::Type{T}) where {T<:Number} =
    projection_matrix(proj, T)

"""
    project_op(observable, projector; dtype=ComplexF64)

Project a square observable down (`P' * O * P`) or up (`P * O * P'`)
depending on which projector dimension matches the observable.
"""
function project_op(observable, proj; dtype::Type=ComplexF64)
    Obs = Matrix(observable)
    ndims(Obs) == 2 && size(Obs, 1) == size(Obs, 2) ||
        throw(ArgumentError("observable must be a square matrix"))
    P = _projector(proj, dtype)
    projected = if size(Obs, 1) == size(P, 1)
        P' * Obs * P
    elseif size(Obs, 1) == size(P, 2)
        P * Obs * P'
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
        result = reduce(
            hcat,
            (V * (exp.((-im * time) .* E) .* coefficients) for time in times),
        )
        return iterate ? (copy(view(result, :, index)) for index in axes(result, 2)) : result
    end

    size(psi, 1) == size(psi, 2) == length(E) ||
        throw(DimensionMismatch("a mixed state must be square and match E"))
    rho_eigen = V' * psi * V
    evolved = [
        begin
            phases = exp.((-im * time) .* E)
            V * (phases .* rho_eigen .* transpose(conj.(phases))) * V'
        end
        for time in times
    ]
    iterate && return (matrix for matrix in evolved)
    return cat(evolved...; dims=3)
end

"""
    ent_entropy(system_state, basis; chain_subsys=nothing, DM=false, ...)

Compatibility entry point for QuSpin's deprecated measurements helper. New
Julia code should prefer `ent_entropy(basis, system_state; ...)`.
"""
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
            results[key] = [
                begin
                    matrix = _observable_matrix(observable, time_values[index])
                    dot(@view(states[:, index]), matrix * @view(states[:, index]))
                end
                for index in axes(states, 2)
            ]
        end
    elseif ndims(states) == 3
        size(states, 3) == length(time_values) ||
            throw(DimensionMismatch("one density-matrix slice is required per time"))
        for (key, observable) in observables
            results[key] = [
                begin
                    matrix = _observable_matrix(observable, time_values[index])
                    tr(@view(states[:, :, index]) * matrix)
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
        ndims(states) == 2 ||
            throw(ArgumentError("time-resolved entropy currently requires pure states"))
        entropy_rows = [
            ent_entropy(basis, @view(states[:, index]); entropy_kwargs...)
            for index in axes(states, 2)
        ]
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
    probabilities = real.(diag(eigenvectors' * state * eigenvectors))
    return probabilities, "DM"
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
    overlaps = abs2.(eigenvectors' * V1)
    weights = reduce(
        hcat,
        begin
            values = distribution(E1, parameter)
            values ./ sum(values)
        end
        for parameter in parameter_values
    )
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
    energies = collect(E2)
    length(energies) == size(V2, 2) ||
        throw(DimensionMismatch("E2 and V2 dimensions do not match"))
    sorted = sort(energies)
    tolerance = 1000eps(float(real(one(eltype(energies)))))
    all(diff(sorted) .> tolerance) ||
        throw(ArgumentError("E2 must be nondegenerate"))
    probabilities, label = _diagonal_probabilities(system_state, V2)
    probabilities = max.(real.(probabilities), eps(Float64))
    requested_observable = !(Obs === nothing || Obs === false)
    (delta_t_Obs || delta_q_Obs) && !requested_observable &&
        throw(ArgumentError("observable fluctuations require Obs"))

    result = Dict{String,Any}()
    divisor = density ? N : 1
    observable_values = nothing
    temporal_variance = nothing
    if requested_observable
        observable = _floquet_matrix(Obs)
        size(observable) == (size(V2, 1), size(V2, 1)) ||
            throw(DimensionMismatch("Obs and V2 dimensions do not match"))
        eigen_observable = V2' * observable * V2
        diagonal_values = real.(diag(eigen_observable))
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
                squared_diagonal = real.(diag(eigen_observable * eigen_observable))
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
        entropy = if alpha == 1
            -sum(probabilities .* log.(probabilities); dims=1)
        else
            log.(sum(probabilities .^ alpha; dims=1)) ./ (1 - alpha)
        end
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
        diagonal_density = V2 * Diagonal(vec(probabilities)) * V2'
        entropy = ent_entropy(
            basis,
            diagonal_density;
            sub_sys_A=subsystem,
            alpha,
        )["Sent_A"]
        key = alpha == 1 ? "Srdm_$label" : "Srdm_Renyi_$label"
        result[key] = entropy
    end

    rho_d && (result["rho_d"] = probabilities)
    return result
end

"""
    evolve(v0, t0, times, f; f_params=(), max_step=0.01, iterate=false, ...)

Integrate a user-defined first-order ODE with a native fourth-order Runge-Kutta
scheme. `f(t, state, f_params...)` must return the state derivative.
"""
function evolve(
    v0::AbstractArray,
    t0::Real,
    times,
    f::Function;
    solver_name=:rk4,
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
    targets = collect(times)
    issorted(targets) || throw(ArgumentError("times must be sorted"))
    state = copy(v0)
    current = float(t0)
    outputs = Any[]
    for target in targets
        target >= current ||
            throw(ArgumentError("times must not precede the current integration time"))
        interval = target - current
        steps = max(1, ceil(Int, abs(interval) / max_step))
        step = interval / steps
        for _ in 1:steps
            k1 = f(current, state, f_params...)
            k2 = f(current + step / 2, state + (step / 2) * k1, f_params...)
            k3 = f(current + step / 2, state + (step / 2) * k2, f_params...)
            k4 = f(current + step, state + step * k3, f_params...)
            state = state + (step / 6) * (k1 + 2k2 + 2k3 + k4)
            current += step
        end
        imag_time && (state ./= norm(state))
        verbose && @info "evolve integrated" time=target
        push!(outputs, copy(state))
    end
    iterate && return (output for output in outputs)
    first(outputs) isa AbstractVector && return reduce(hcat, outputs)
    return cat(outputs...; dims=ndims(first(outputs)) + 1)
end

"""
    ExpmMultiplyParallel(A, a=1; dtype=nothing, copy=false)

Native Julia representation of the linear action `exp(a*A) * v`.  The current
implementation favors a small, exact public contract; specialized sparse
Krylov kernels can replace the dense backend without changing this API.
"""
mutable struct ExpmMultiplyParallel{M}
    A::M
    a::Number
    dtype::DataType
end

function ExpmMultiplyParallel(
    A,
    a::Number=1.0;
    dtype::Union{Nothing,Type}=nothing,
    copy::Bool=false,
)
    ndims(A) == 2 && size(A, 1) == size(A, 2) ||
        throw(ArgumentError("A must be a square matrix"))
    T = dtype === nothing ? promote_type(eltype(A), typeof(a), Float64) : dtype
    T <: Number || throw(ArgumentError("dtype must be numeric"))
    stored = copy ? Base.copy(A) : A
    return ExpmMultiplyParallel{typeof(stored)}(stored, a, T)
end

const expm_multiply_parallel = ExpmMultiplyParallel

"""
    set_a!(operator::ExpmMultiplyParallel, a; dtype=nothing)

Update the scalar multiplying the exponential generator.
"""
function set_a!(
    operator::ExpmMultiplyParallel,
    a::Number;
    dtype::Union{Nothing,Type}=nothing,
)
    operator.a = a
    operator.dtype = dtype === nothing ?
        promote_type(eltype(operator.A), typeof(a), Float64) :
        dtype
    return operator
end

"""
    apply(operator::ExpmMultiplyParallel, v; work_array=nothing,
          overwrite_v=false, tol=nothing)

Apply the matrix exponential to one vector or a batch of column vectors.
"""
function apply(
    operator::ExpmMultiplyParallel,
    v::AbstractVecOrMat;
    work_array=nothing,
    overwrite_v::Bool=false,
    tol=nothing,
)
    size(v, 1) == size(operator.A, 2) ||
        throw(DimensionMismatch("A and v dimensions do not match"))
    if work_array !== nothing
        length(work_array) == 2length(v) ||
            throw(DimensionMismatch("work_array must contain twice as many elements as v"))
    end
    T = promote_type(operator.dtype, eltype(v))
    result = _krylov_expmv(
        operator.A,
        MatrixOrVector(v, T),
        convert(T, operator.a);
        tol=tol === nothing ? 1e-12 : tol,
    )
    overwrite_v || return result
    eltype(v) == eltype(result) ||
        throw(ArgumentError("overwrite_v requires an input with the output dtype"))
    copyto!(v, result)
    return v
end

MatrixOrVector(v::AbstractVector, ::Type{T}) where {T} = Vector{T}(v)
MatrixOrVector(v::AbstractMatrix, ::Type{T}) where {T} = Matrix{T}(v)

Base.:*(operator::ExpmMultiplyParallel, v::AbstractVecOrMat) = apply(operator, v)

struct StroboscopicTimes{T}
    inds::Vector{Int}
    vals::Vector{T}
end

struct FloquetTimeSegment{T}
    vals::Vector{T}
    len::Int
    shape::Tuple{Int}
    i::T
    f::T
    tot::T
    strobo::StroboscopicTimes{T}
end

function _time_segment(values::AbstractVector{T}, stride::Int) where {T}
    vals = collect(values)
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

_dict_get(dictionary::AbstractDict, key::Symbol, default=nothing) =
    get(dictionary, key, get(dictionary, String(key), default))
_dict_has(dictionary::AbstractDict, key::Symbol) =
    haskey(dictionary, key) || haskey(dictionary, String(key))
_floquet_matrix(H::Hamiltonian, time=0.0) =
    Matrix{ComplexF64}(toarray(H; time))
_floquet_matrix(H, time=0.0) =
    Matrix{ComplexF64}(H isa Function ? H(time) : H)

function _step_unitary(matrices, durations)
    length(matrices) == length(durations) ||
        throw(DimensionMismatch("Hamiltonian and duration lists must have equal lengths"))
    isempty(matrices) && throw(ArgumentError("at least one evolution step is required"))
    first_matrix = _floquet_matrix(first(matrices))
    size(first_matrix, 1) == size(first_matrix, 2) ||
        throw(ArgumentError("Hamiltonians must be square"))
    unitary = Matrix{ComplexF64}(I, size(first_matrix)...)
    for (H, duration) in zip(matrices, durations)
        operator = H isa Hamiltonian && isempty(H.dynamic_terms) ?
            H :
            _floquet_matrix(H)
        size(operator) == size(first_matrix) ||
            throw(DimensionMismatch("all Hamiltonians must have the same shape"))
        unitary = _krylov_expmv(operator, unitary, -im * duration)
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
        T = float(_dict_get(evolution, :T, sum(dt_list)))
        T, _step_unitary(H_list, dt_list)
    elseif _dict_has(evolution, :H) &&
           _dict_has(evolution, :t_list) &&
           _dict_has(evolution, :dt_list)
        H = _dict_get(evolution, :H)
        t_list = collect(_dict_get(evolution, :t_list))
        dt_list = collect(_dict_get(evolution, :dt_list))
        length(t_list) == length(dt_list) ||
            throw(DimensionMismatch("t_list and dt_list must have equal lengths"))
        matrices = [_floquet_matrix(H, time) for time in t_list]
        T = float(_dict_get(evolution, :T, sum(dt_list)))
        T, _step_unitary(matrices, dt_list)
    elseif _dict_has(evolution, :H) && _dict_has(evolution, :T)
        requested_H = _dict_get(evolution, :H)
        H = requested_H isa Hamiltonian && isempty(requested_H.dynamic_terms) ?
            requested_H :
            _floquet_matrix(requested_H)
        T = float(_dict_get(evolution, :T))
        identity = Matrix{ComplexF64}(I, size(H)...)
        T, _krylov_expmv(H, identity, -im * T)
    else
        throw(ArgumentError("unsupported Floquet evolution dictionary"))
    end
    period > 0 || throw(ArgumentError("the Floquet period must be positive"))

    decomposition = eigen(unitary)
    phases = ComplexF64.(decomposition.values)
    vectors = ComplexF64.(decomposition.vectors)
    energies = real.((im / period) .* log.(phases))
    order = sortperm(energies)
    energies = Float64.(energies[order])
    phases = phases[order]
    vectors = vectors[:, order]
    if force_ONB
        vectors = Matrix(qr(vectors).Q)
    end

    effective = HF ? ComplexF64.((im / period) .* log(unitary)) : nothing
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

function _dense_block_diagonal(matrices)
    isempty(matrices) && throw(ArgumentError("at least one block is required"))
    T = promote_type(map(eltype, matrices)...)
    rows = sum(size(matrix, 1) for matrix in matrices)
    cols = sum(size(matrix, 2) for matrix in matrices)
    result = zeros(T, rows, cols)
    row = 1
    column = 1
    for matrix in matrices
        next_row = row + size(matrix, 1) - 1
        next_column = column + size(matrix, 2) - 1
        result[row:next_row, column:next_column] .= matrix
        row = next_row + 1
        column = next_column + 1
    end
    return result
end

"""
    block_diag_hamiltonian(blocks, static, dynamic, basis_con, basis_args,
                           dtype; get_proj=true, ...)

Build symmetry-sector bases, their projectors, and the corresponding dense
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
    isempty(dynamic) ||
        throw(ArgumentError("dynamic block Hamiltonians are not yet supported"))
    terms = _normalize_block_terms(static)
    bases = [
        _construct_block_basis(basis_con, basis_args, basis_kwargs, block)
        for block in blocks
    ]
    hamiltonians = [Hamiltonian(basis, terms) for basis in bases]
    matrix = Matrix{dtype}(_dense_block_diagonal(Matrix.(hamiltonians)))
    get_proj || return matrix
    projectors = [projection_matrix(basis, dtype) for basis in bases]
    return reduce(hcat, projectors), matrix
end

mutable struct BlockOps
    basis_dict::Dict{String,Any}
    H_dict::Dict{String,Any}
    P_dict::Dict{String,Any}
    dtype::DataType
    static::Vector{OperatorTerm}
    dynamic::Vector{Any}
    save_previous_data::Bool
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
    isempty(dynamic) ||
        throw(ArgumentError("dynamic block Hamiltonians are not yet supported"))
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
    )
    compute_all_blocks && compute_all_blocks!(operator)
    return operator
end

function compute_all_blocks!(operator::BlockOps)
    for (key, basis) in operator.basis_dict
        operator.P_dict[key] = projection_matrix(basis, operator.dtype)
        operator.H_dict[key] = Hamiltonian(basis, operator.static)
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
        basis = _construct_block_basis(basis_con, basis_args, Dict(), block)
        length(basis) > 0 && (operator.basis_dict[key] = basis)
    end
    compute_all_blocks && compute_all_blocks!(operator)
    return operator
end

function _block_data(operator::BlockOps, key, basis)
    projector = get!(
        () -> projection_matrix(basis, operator.dtype),
        operator.P_dict,
        key,
    )
    hamiltonian = get!(
        () -> Hamiltonian(basis, operator.static),
        operator.H_dict,
        key,
    )
    return projector, hamiltonian
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
    imag_time && throw(ArgumentError("imaginary-time block evolution is unsupported"))
    n_jobs > 0 || throw(ArgumentError("n_jobs must be positive"))
    targets = collect(times)
    result = zeros(
        promote_type(ComplexF64, eltype(psi_0)),
        length(psi_0),
        length(targets),
    )
    active = false
    for (key, basis) in operator.basis_dict
        projector, hamiltonian = _block_data(operator, key, basis)
        projected = projector' * psi_0
        norm(projected) > 1000eps(Float64) || continue
        active = true
        for (column, time) in pairs(targets)
            result[:, column] .+= projector * _krylov_expmv(
                hamiltonian,
                projected,
                -im * (time - t0),
            )
        end
    end
    active || throw(ArgumentError("initial state has no projection onto the selected blocks"))
    iterate && return (copy(view(result, :, index)) for index in axes(result, 2))
    return result
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
    scales = start === nothing && stop === nothing ?
        [one(float(real(a)))] :
        collect(range(start, stop; length=num + (!endpoint)))
    !endpoint && pop!(scales)
    result = zeros(
        promote_type(ComplexF64, eltype(psi_0), typeof(a)),
        length(psi_0),
        length(scales),
    )
    for (key, basis) in operator.basis_dict
        projector, hamiltonian = _block_data(operator, key, basis)
        projected = projector' * psi_0
        norm(projected) > 1000eps(Float64) || continue
        generator = shift === nothing ?
            hamiltonian :
            Matrix(hamiltonian) + shift * I
        for (column, scale) in pairs(scales)
            result[:, column] .+= projector * _krylov_expmv(
                generator,
                projected,
                scale * a,
            )
        end
    end
    output = length(scales) == 1 ? vec(result) : result
    iterate && return (copy(view(result, :, index)) for index in axes(result, 2))
    return output
end

function _lanczos_action(A, vector)
    result = A * vector
    result isa AbstractVector ||
        throw(ArgumentError("A must act on a vector and return a vector"))
    return result
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
    norm(v0) > 0 || throw(ArgumentError("initial Lanczos vector must be nonzero"))

    first_action = _lanczos_action(A, v0)
    T = promote_type(eltype(v0), eltype(first_action))
    RT = typeof(real(zero(T)))
    tolerance = eps === nothing ? Base.eps(RT) : convert(RT, eps)
    Q_T = out === nothing ? Matrix{T}(undef, m, n) : out
    size(Q_T) == (m, n) ||
        throw(DimensionMismatch("out must have shape (m, length(v0))"))

    q = Vector{T}(v0)
    q ./= norm(q)
    q_previous = zeros(T, n)
    beta_previous = zero(RT)
    alphas = RT[]
    betas = RT[]
    used = 0

    for index in 1:m
        Q_T[index, :] = q
        used = index
        residual = Vector{T}(_lanczos_action(A, q))
        index > 1 && (residual .-= beta_previous .* q_previous)
        alpha = real(dot(q, residual))
        residual .-= alpha .* q
        if full_ortho
            for previous in 1:index
                q_basis = @view Q_T[previous, :]
                residual .-= dot(q_basis, residual) .* q_basis
            end
        end
        push!(alphas, alpha)
        index == m && break
        beta = norm(residual)
        beta <= tolerance && break
        push!(betas, beta)
        q_previous, q = q, residual ./ beta
        beta_previous = beta
    end

    tridiagonal = SymTridiagonal(alphas, betas)
    decomposition = eigen(tridiagonal)
    return decomposition.values, decomposition.vectors, Q_T[1:used, :]
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
    initial = copy_v0 ? copy(v0) : v0
    operator = copy_A ? deepcopy(A) : A
    E, V, Q_T = lanczos_full(operator, initial, m; eps)
    return_vec_iter || return E, V
    return E, V, (copy(row) for row in eachrow(Q_T))
end

"""Compute a coefficient-weighted linear combination of Lanczos rows."""
function lin_comb_Q_T(coeff, Q_T; out=nothing)
    rows = Q_T isa AbstractMatrix ? Q_T : reduce(vcat, (permutedims(row) for row in Q_T))
    length(coeff) == size(rows, 1) ||
        throw(DimensionMismatch("one coefficient is required per Lanczos vector"))
    result = vec(transpose(coeff) * rows)
    out === nothing && return result
    axes(out) == axes(result) ||
        throw(DimensionMismatch("out must match a Lanczos vector"))
    copyto!(out, result)
    return out
end

"""Apply a matrix exponential to the Lanczos starting vector."""
function expm_lanczos(E, V, Q_T; a::Number=1, out=nothing)
    length(E) == size(V, 1) == size(V, 2) ||
        throw(DimensionMismatch("E and V dimensions must agree"))
    coefficients = V * (exp.(a .* E) .* V[1, :])
    return lin_comb_Q_T(coefficients, Q_T; out)
end

function _lanczos_rows(Q_T)
    return Q_T isa AbstractMatrix ?
        Q_T :
        reduce(vcat, (permutedims(row) for row in Q_T))
end

_temperature_output(values, beta) = beta isa Number ? only(values) : values

"""
    ftlm_static_iteration(observables, E, V, Q_T; beta=0)

One finite-temperature Lanczos trace-estimation iteration. Returns a dictionary
of unnormalized observable estimates and the corresponding identity estimate.
"""
function ftlm_static_iteration(observables::AbstractDict, E, V, Q_T; beta=0)
    rows = _lanczos_rows(Q_T)
    m = length(E)
    size(V) == (m, m) && size(rows, 1) == m ||
        throw(DimensionMismatch("Lanczos inputs must share the same Krylov dimension"))
    betas = beta isa Number ? [beta] : collect(beta)
    probabilities = exp.(-E .* transpose(betas))
    coefficients = V * (V[1, :] .* probabilities)
    initial = copy(@view rows[1, :])

    results = Dict{Any,Any}()
    for (key, observable) in observables
        applied = observable * initial
        overlaps = [dot(@view(rows[index, :]), applied) for index in 1:m]
        values = vec(transpose(overlaps) * coefficients)
        results[key] = _temperature_output(values, beta)
    end
    identity = _temperature_output(vec(coefficients[1, :]), beta)
    return results, identity
end

"""
    ltlm_static_iteration(observables, E, V, Q_T; beta=0)

One symmetric low-temperature Lanczos trace-estimation iteration.
"""
function ltlm_static_iteration(observables::AbstractDict, E, V, Q_T; beta=0)
    rows = _lanczos_rows(Q_T)
    m = length(E)
    size(V) == (m, m) && size(rows, 1) == m ||
        throw(DimensionMismatch("Lanczos inputs must share the same Krylov dimension"))
    betas = beta isa Number ? [beta] : collect(beta)
    weights = V[1, :] .* exp.(-E .* transpose(betas) ./ 2)
    results = Dict{Any,Any}()

    for (key, observable) in observables
        first_applied = observable * collect(@view rows[1, :])
        MT = promote_type(eltype(rows), eltype(first_applied))
        matrix_elements = Matrix{MT}(undef, m, m)
        for row in 1:m
            applied = row == 1 ?
                first_applied :
                observable * collect(@view rows[row, :])
            for column in 1:m
                matrix_elements[row, column] = dot(@view(rows[column, :]), applied)
            end
        end
        projected = transpose(V) * matrix_elements * V
        values = [
            dot(@view(weights[:, index]), projected * @view(weights[:, index]))
            for index in axes(weights, 2)
        ]
        results[key] = _temperature_output(values, beta)
    end

    identity_values = [
        sum(V[1, :] .^ 2 .* exp.(-betas[index] .* E))
        for index in eachindex(betas)
    ]
    return results, _temperature_output(identity_values, beta)
end

end
