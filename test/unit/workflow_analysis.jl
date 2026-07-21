using Test
using LinearAlgebra
using SparseArrays
using QuSpin

@testset "Constraint state generation" begin
    N = 6
    states = constraint_states(
        N;
        prefix_allowed=(occupations, site) ->
            site == 1 || occupations[site - 1] + occupations[site] <= 1,
        state_allowed=occupations -> occupations[1] + occupations[end] <= 1,
    )
    @test length(states) == 18
    @test issorted(states)
    @test all(states) do state
        bits = [Int((state >> (site - 1)) & 1) for site in 1:N]
        all(bits[site] + bits[mod1(site + 1, N)] <= 1 for site in 1:N)
    end

    basis = UserBasis(
        UInt64,
        N,
        Dict('n' => [0.0 0.0; 0.0 1.0]);
        states,
        allowed_ops=('n',),
    )
    @test length(basis) == 18
    @test basis.states == states
end

@testset "Subspace fidelity and parallel transport" begin
    left = ComplexF64[
        1 0
        0 1
        0 0
        0 0
    ]
    rotation = ComplexF64[
        1 im
        im 1
    ] / sqrt(2)
    right = left * rotation
    @test subspace_fidelity(left, right) ≈ 1 atol=2e-15
    @test subspace_fidelity(left, right; aggregate=:all) ≈ ones(2) atol=2e-15
    @test subspace_fidelity(left[:, 1], right[:, 1]) ≈ inv(sqrt(2)) atol=2e-15

    tracked = track_eigenspaces([left, right])
    @test tracked.fidelities ≈ [1.0] atol=2e-15
    @test tracked.spaces[2] ≈ left atol=2e-15

    orthogonal = ComplexF64[
        0 0
        0 0
        1 0
        0 1
    ]
    @test subspace_fidelity(left, orthogonal) ≈ 0 atol=2e-15
end

@testset "Dynamical and frequency-domain response" begin
    gap = 1.7
    H = Diagonal([0.0, gap])
    X = ComplexF64[0 1; 1 0]
    state = ComplexF64[1, 0]
    times = [0.0, 0.2, 0.9]
    correlation = dynamical_correlator(H, state, X, X, times)
    @test correlation ≈ exp.(-im .* gap .* times) atol=2e-12

    source_state = ComplexF64[1]
    transition = reshape(ComplexF64[0, 1], 2, 1)
    frequencies = [gap - 0.3, gap, gap + 0.3]
    broadening = 0.08
    expected = @. broadening /
        (pi * ((frequencies - gap)^2 + broadening^2))
    for method in (:lehmann, :resolvent, :krylov)
        spectrum = spectral_function(
            H,
            source_state,
            transition,
            frequencies;
            broadening,
            method,
            krylov_dim=1,
        )
        @test spectrum ≈ expected atol=3e-12 rtol=3e-12
    end
    @test spectral_function(
        H,
        source_state,
        zeros(ComplexF64, 2, 1),
        frequencies,
    ) == zeros(length(frequencies))
end

@testset "Matrix-free Lindblad evolution" begin
    decay = 0.7
    H = zeros(ComplexF64, 2, 2)
    jump = sqrt(decay) .* ComplexF64[0 1; 0 0]
    generator = LindbladGenerator(H, [jump])
    excited = ComplexF64[0 0; 0 1]

    action = reshape(generator * vec(excited), 2, 2)
    @test action ≈ ComplexF64[decay 0; 0 -decay] atol=2e-15
    @test Matrix(generator) * vec(excited) ≈ vec(action) atol=2e-15

    times = [0.0, 0.4, 1.2]
    states = evolve(
        generator,
        excited,
        0.0,
        times;
        max_step=0.02,
        rtol=1e-10,
        atol=1e-12,
    )
    for (index, time) in pairs(times)
        rho = @view states[:, :, index]
        population = exp(-decay * time)
        @test rho ≈ ComplexF64[1 - population 0; 0 population] atol=2e-9
        @test tr(rho) ≈ 1 atol=2e-10
        @test rho ≈ rho' atol=2e-10
        @test minimum(eigvals(Hermitian(Matrix(rho)))) >= -2e-10
    end
end
