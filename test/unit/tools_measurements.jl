@testset "entanglement and time-dependent measurement properties" begin
    basis = SpinBasis1D(4)
    ghz = zeros(ComplexF64, length(basis))
    ghz[state_index(basis, 0)] = inv(sqrt(2))
    ghz[state_index(basis, 15)] = inv(sqrt(2))

    result = ent_entropy(
        basis,
        ghz;
        return_rdm=:both,
        return_rdm_EVs=true,
    )
    @test result["Sent_A"] ≈ 0.34657359027997265 atol=2e-16
    @test result["Sent_B"] ≈ 0.34657359027997265 atol=2e-16
    @test sort(result["p_A"]) ≈ [0.0, 0.0, 0.5, 0.5] atol=3e-16
    @test result["rdm_A"] ≈ Diagonal([0.5, 0.0, 0.0, 0.5]) atol=3e-16
    @test result["rdm_B"] ≈ result["rdm_A"] atol=3e-16

    fixed = SpinBasis1D(4; nup=2)
    pair = zeros(ComplexF64, length(fixed))
    pair[state_index(fixed, 3)] = inv(sqrt(2))
    pair[state_index(fixed, 12)] = inv(sqrt(2))
    @test ent_entropy(fixed, pair)["Sent_A"] ≈
        0.34657359027997265 atol=2e-16
    @test tr(partial_trace(fixed, pair; return_rdm=:A)) ≈ 1.0 atol=3e-16

    old_result = ent_entropy(ghz, basis; DM=:both)
    @test old_result["Sent"] == old_result["Sent_A"]
    @test old_result["DM_chain_subsys"] == old_result["rdm_A"]
    @test old_result["DM_other_subsys"] == old_result["rdm_B"]

    A = [
        1.0 0.2 0.0 0.0
        0.2 2.0 0.3 0.0
        0.0 0.3 3.0 0.4
        0.0 0.0 0.4 4.0
    ]
    B = Diagonal([2.0, -1.0, 0.5, 3.0])
    E, V = eigen(Symmetric(A))
    times = [0.0, 0.25, 1.0]
    observations = obs_vs_time(
        (ComplexF64[1, 0, 0, 0], E, V),
        times,
        Dict("A" => A, "B" => B);
        return_state=true,
        Sent_args=Dict(:basis => SpinBasis1D(2)),
    )
    @test observations["A"] ≈ ComplexF64[
        1.0 + 0.0im,
        1.0 - 2.17756884e-17im,
        1.0 + 5.34647196e-18im,
    ] atol=1e-15
    @test observations["B"] ≈ ComplexF64[
        2.0 + 0.0im,
        1.99255385 + 5.90437098e-18im,
        1.89288098 + 2.77433877e-17im,
    ] atol=6e-9
    @test size(observations["psi_t"]) == (4, 3)
    @test length(observations["Sent_time"]["Sent_A"]) == 3
    dynamic = obs_vs_time(
        observations["psi_t"],
        times,
        Dict("scaled_identity" => time -> time * Matrix{Float64}(I, 4, 4));
        enforce_pure=true,
    )
    @test real.(dynamic["scaled_identity"]) ≈ times atol=2e-15
end

@testset "batched pure-state entropy" begin
    basis = SpinBasis1D(3; pauli=false)
    ghz = zeros(ComplexF64, 8)
    ghz[[1, 8]] .= inv(sqrt(2))
    product = ComplexF64[1, zeros(7)...]
    states = hcat(ghz, product)
    result = ent_entropy(
        basis,
        states;
        sub_sys_A=[1],
        density=false,
        enforce_pure=true,
        return_rdm=:A,
    )
    independent = [
        ent_entropy(
            basis,
            @view(states[:, index]);
            sub_sys_A=[1],
            density=false,
        )["Sent_A"]
        for index in axes(states, 2)
    ]
    @test result["Sent_A"] ≈ independent atol=4e-14
    @test size(result["rdm_A"]) == (2, 2, 2)

    mixed = cat(
        ghz * ghz',
        product * product';
        dims=3,
    )
    measured = obs_vs_time(
        mixed,
        [0.0, 0.4],
        Dict("I" => Matrix{ComplexF64}(I, 8, 8));
        Sent_args=Dict(
            :basis => basis,
            :sub_sys_A => [1],
            :density => false,
        ),
    )
    @test measured["Sent_time"]["Sent_A"] ≈ independent atol=4e-14
end
