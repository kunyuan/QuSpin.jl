using Test
using LinearAlgebra
using SparseArrays
using QuSpin

@testset "Public API namespaces" begin
    @test isdefined(QuSpin, :Basis)
    @test isdefined(QuSpin, :Operators)
    @test isdefined(QuSpin, :Tools)
    @test QuSpin.Basis.SpinBasis1D === SpinBasis1D
    @test QuSpin.Operators.Hamiltonian === Hamiltonian
    @test isempty(detect_ambiguities(QuSpin; recursive=true))
end

@testset "Wide basis integers" begin
    wide_value = (BigInt(1) << 200) + 3
    encoded = python_int_to_basis_int(wide_value)
    @test encoded isa UInt256
    @test basis_int_to_python_int(encoded) == wide_value
    @test python_int_to_basis_int(typemax(UInt32)) isa UInt32
    @test python_int_to_basis_int(BigInt(typemax(UInt32)) + 1) isa UInt64
    @test get_basis_type(32, nothing, 2) === UInt32
    @test get_basis_type(64, nothing, 2) === UInt64
    @test get_basis_type(100, 1, 2) === UInt256
    @test get_basis_type(300, nothing, 2) === UInt1024
    @test ~UInt256(0) == typemax(UInt256)
    @test (UInt256(7) & UInt256(3)) == 3
    @test (UInt256(4) | UInt256(3)) == 7
    @test xor(UInt256(7), UInt256(3)) == 4
    @test UInt256(7) == BigInt(7) == UInt256(7)
    @test_throws ArgumentError UInt256(7) & UInt1024(3)
    @test (UInt256(3) << 4) == 48
    @test (UInt256(48) >> 4) == 3
    @test_throws ArgumentError python_int_to_basis_int(-1)
    @test_throws InexactError UInt256(BigInt(1) << 256)
end

@testset "Basis array and bitwise helpers" begin
    @test basis_zeros((2, 3), UInt256) == fill(UInt256(0), 2, 3)
    @test basis_ones(3, UInt1024) == fill(UInt1024(1), 3)

    left = UInt32[0x0f, 0xf0, 0xaa]
    right = UInt32[0x33, 0x55, 0xff]
    @test bitwise_and(left, right) == UInt32[0x03, 0x50, 0xaa]
    @test bitwise_or(left, right) == UInt32[0x3f, 0xf5, 0xff]
    @test bitwise_xor(left, right) == UInt32[0x3c, 0xa5, 0x55]
    @test bitwise_not(UInt8[0x00, 0xff]) == UInt8[0xff, 0x00]
    @test bitwise_leftshift(UInt32[1, 2, 3], [1, 2, 3]) == UInt32[2, 8, 24]
    @test bitwise_rightshift(UInt32[8, 16, 32], [1, 2, 3]) == UInt32[4, 4, 4]

    out = UInt32[100, 100, 100]
    @test bitwise_and(left, right; out, where=[true, false, true]) === out
    @test out == UInt32[0x03, 100, 0xaa]
    @test bitwise_or(left, right; where=[false, true, false]) ==
        UInt32[0x00, 0xf5, 0x00]
end

@testset "Photon helpers" begin
    @test coherent_state(0.5, 5) ≈ [
        0.8824969025845955,
        0.4412484512922977,
        0.15600488604842286,
        0.045034731477476914,
        0.011258682869369227,
    ] atol=2e-17 rtol=2e-15
    @test coherent_state(0.0, 4) == [1.0, 0.0, 0.0, 0.0]
    @test coherent_state(-0.5, 4) ≈
        [0.8824969025845955, -0.4412484512922977, 0.15600488604842286, -0.045034731477476914]
    @test coherent_state(0.5 + 0.25im, 3; dtype=ComplexF64) ≈ [
        0.8553453273074225 + 0.0im,
        0.42767266365371126 + 0.21383633182685563im,
        0.11340384022411978 + 0.15120512029882638im,
    ] atol=2e-16 rtol=2e-15
    @test photon_Hspace_dim(4, nothing, 3) == 64
    @test photon_Hspace_dim(4, 0, nothing) == 1
    @test photon_Hspace_dim(4, 1, nothing) == 5
    @test photon_Hspace_dim(4, 2, nothing) == 11
    @test photon_Hspace_dim(4, 4, nothing) == 16
    @test_throws ArgumentError coherent_state(0.5, 0)
    @test_throws ArgumentError photon_Hspace_dim(4, nothing, nothing)
end

@testset "Tools.misc equivalents" begin
    binary = UInt8[
        0 0 0 0
        0 0 0 1
        0 0 1 0
        0 0 1 1
        1 0 1 0
    ]
    encoded = UInt32[0, 1, 2, 3, 10]
    @test ints_to_array(encoded, 4) == binary
    @test array_to_ints(binary) == encoded
    @test array_to_ints(binary, UInt256) == UInt256.(encoded)
    wide_encoded = UInt256[(UInt256(1) << 200) | UInt256(3)]
    @test array_to_ints(ints_to_array(wide_encoded, 201), UInt256) ==
        wide_encoded

    @test kl_div([0.25, 0.75], [0.5, 0.5]) ≈ 0.13081203594113697
    @test_throws DimensionMismatch kl_div([0.5, 0.5], [1.0])
    @test_throws ArgumentError kl_div([0.0, 1.0], [0.5, 0.5])
    @test_throws ArgumentError kl_div([0.4, 0.4], [0.5, 0.5])

    @test mean_level_spacing([0.0, 1.0, 3.0, 6.0]) ≈ 7 / 12
    @test isnan(mean_level_spacing([0.0, 1.0, 1.0, 2.0]; verbose=false))
    @test_throws ArgumentError mean_level_spacing([0.0, 2.0, 1.0])

    A = [1.0 2.0; 3.0 4.0]
    v = [2.0, -1.0]
    @test matvec(A, v; a=2.0) == [0.0, 4.0]
    selected_matvec = get_matvec_function(A)
    @test selected_matvec(A, v; a=2.0) == [0.0, 4.0]
    out = [10.0, 20.0]
    @test matvec(A, v; a=2.0, out) === out
    @test out == [10.0, 24.0]
    @test matvec(A, v; a=2.0, out, overwrite_out=true) === out
    @test out == [0.0, 4.0]

    P = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    @test project_op(Diagonal([1.0, 2.0, 3.0]), P)["Proj_Obs"] ==
        ComplexF64[1 0; 0 2]
    @test project_op(Diagonal([4.0, 5.0]), P)["Proj_Obs"] ==
        ComplexF64[4 0 0; 0 5 0; 0 0 0]
end

@testset "Operator algebra and Krylov tools" begin
    A = [
        1.0 0.2 0.0 0.0
        0.2 2.0 0.3 0.0
        0.0 0.3 3.0 0.4
        0.0 0.0 0.4 4.0
    ]
    B = Diagonal([2.0, -1.0, 0.5, 3.0])
    @test commutator(A, B) == A * B - B * A
    @test anti_commutator(A, B) == A * B + B * A

    v0 = [1.0, 2.0, -1.0, 0.5]
    E, V, Q_T = lanczos_full(A, v0, 3)
    @test E ≈ [1.0981001507517356, 1.9892033688829862, 3.8667165876454432] atol=5e-15
    @test Q_T * Q_T' ≈ Matrix(I, 3, 3) atol=2e-15
    @test lin_comb_Q_T([1.0, -2.0, 0.5], Q_T) ≈
        [1.9862687479070797, 0.6259141580861511, 0.2815640364800571, -0.91306605519864] atol=2e-15
    @test expm_lanczos(E, V, Q_T; a=-0.25) ≈
        [0.28370463815222685, 0.49009609392027137, -0.2301972310236773, 0.09222361279644782] atol=5e-15

    E_iter, V_iter, Q_iter = lanczos_iter(A, v0, 3)
    @test E_iter ≈ E atol=1e-14
    @test abs.(V_iter) ≈ abs.(V) atol=1e-14
    @test reduce(vcat, permutedims.(collect(Q_iter))) ≈ Q_T atol=1e-14

    exact_E, exact_V = eigen(Symmetric(A))
    psi = ComplexF64[1, 0, 0, 0]
    times = [0.0, 0.25, 1.0]
    psi_t = ed_state_vs_time(psi, exact_E, exact_V, times)
    @test psi_t[:, 1] ≈ psi atol=5e-16
    @test all(isapprox(norm(psi_t[:, t]), 1.0; atol=5e-15) for t in axes(psi_t, 2))
    @test collect(ed_state_vs_time(psi, exact_E, exact_V, times; iterate=true)) ≈
        collect(eachcol(psi_t)) atol=5e-15

    observables = Dict("A" => A, "B" => B)
    beta = [0.0, 0.5, 2.0]
    ftlm, ftlm_identity = ftlm_static_iteration(observables, E, V, Q_T; beta)
    ltlm, ltlm_identity = ltlm_static_iteration(observables, E, V, Q_T; beta)
    @test ftlm_identity ≈ [1.0, 0.38217846291319085, 0.02560632611811718] atol=5e-16
    @test ftlm["B"] ≈ [-0.12, -0.02325674028781951, 0.01521990440418417] atol=8e-16
    @test ltlm_identity ≈ ftlm_identity atol=5e-16
    @test ltlm["B"] ≈ [-0.12, -0.02720657100056404, 0.01224408682892336] atol=8e-16

    ode = (time, state, frequency) -> -im * frequency .* state
    ode_times = [0.0, 0.2, 1.0]
    ode_states = evolve(
        ComplexF64[1],
        0.0,
        ode_times,
        ode;
        f_params=(2.0,),
        max_step=0.002,
    )
    @test vec(ode_states) ≈ exp.(-2im .* ode_times) atol=1e-11
    @test collect(
        evolve(
            ComplexF64[1],
            0.0,
            ode_times,
            ode;
            f_params=(2.0,),
            max_step=0.002,
            iterate=true,
        ),
    ) ≈ collect(eachcol(ode_states)) atol=2e-12
end

@testset "Parallel exponential action protocol" begin
    generator = [0.0 -1.0; 1.0 0.0]
    action = ExpmMultiplyParallel(generator, 0.25)
    @test action.A === generator
    @test action.a == 0.25
    @test expm_multiply_parallel === ExpmMultiplyParallel
    @test apply(action, [1.0, 0.0]) ≈ [cos(0.25), sin(0.25)] atol=3e-16
    batch = [1.0 0.0; 0.0 1.0]
    @test action * batch ≈ exp(0.25 .* generator) * batch atol=3e-16
    destination = ComplexF64[1, 0]
    set_a!(action, -0.5im)
    @test apply(action, destination; overwrite_v=true) === destination
    @test destination ≈ exp((-0.5im) .* generator) * ComplexF64[1, 0] atol=5e-16
    @test_throws DimensionMismatch apply(action, ones(3))
    @test_throws DimensionMismatch apply(action, ones(2); work_array=zeros(3))
end

@testset "Floquet time-vector protocol" begin
    times = FloquetTimeVector(2π, 2; len_T=4, N_up=1, N_down=1)
    @test times.N == 4
    @test times.T ≈ 1.0
    @test times.dt ≈ 0.25
    @test times.vals ≈ collect(-1.0:0.25:3.0)
    @test length(times) == 17
    @test size(times) == (17,)
    @test times.i ≈ -1.0
    @test times.f ≈ 3.0
    @test times.tot ≈ 4.0
    @test times.strobo.inds == [1, 5, 9, 13, 17]
    @test times.strobo.vals ≈ [-1.0, 0.0, 1.0, 2.0, 3.0]
    @test times.up.vals ≈ [-1.0, -0.75, -0.5, -0.25]
    @test times.constant.vals ≈ collect(0.0:0.25:2.0)
    @test times.down.vals ≈ collect(2.25:0.25:3.0)
    @test getproperty(times, :const) === times.constant
    @test get_coordinates(times, 7) == (2, 3)
    @test collect(times) == times.vals
    @test times[7] == 0.5
    @test times * 2 == times.vals * 2
    @test times / 2 == times.vals / 2
end

@testset "Floquet spectrum protocol" begin
    Z = ComplexF64[1 0; 0 -1]
    spectrum = Floquet(
        Dict(:H => Z, :T => 0.5);
        HF=true,
        UF=true,
        thetaF=true,
        VF=true,
    )
    @test spectrum.T == 0.5
    @test spectrum.EF ≈ [-1.0, 1.0] atol=2e-16
    @test spectrum.UF ≈ exp((-0.5im) .* Z) atol=3e-16
    @test spectrum.HF ≈ Z atol=2e-15
    @test spectrum.thetaF ≈
        ComplexF64[exp(0.5im), exp(-0.5im)] atol=3e-16
    @test spectrum.VF' * spectrum.VF ≈ Matrix{ComplexF64}(I, 2, 2) atol=2e-16

    X = ComplexF64[0 1; 1 0]
    expected = exp((-0.3im) .* X) * exp((-0.2im) .* Z)
    stepped = Floquet(
        Dict("H_list" => [Z, X], "dt_list" => [0.2, 0.3]);
        UF=true,
    )
    @test stepped.T == 0.5
    @test stepped.UF ≈ expected atol=4e-16
    @test stepped.HF === nothing
    @test stepped.VF === nothing

    callable = time -> time < 0.5 ? Z : X
    sampled = Floquet(
        Dict(:H => callable, :t_list => [0.0, 0.75], :dt_list => [0.2, 0.3]);
        UF=true,
    )
    @test sampled.UF ≈ expected atol=4e-16
    @test_throws ArgumentError Floquet(Dict(:H => Z))
end

@testset "Symmetry block tools" begin
    terms = [OperatorTerm("z", [(1.0, 1), (0.5, 2)])]
    blocks = [Dict(:nup => sector) for sector in 0:2]
    P, block_H = block_diag_hamiltonian(
        blocks,
        terms,
        Any[],
        SpinBasis1D,
        (2,),
        ComplexF64;
        basis_kwargs=Dict(:pauli => false),
    )
    full_basis = SpinBasis1D(2; pauli=false)
    full_H = Hamiltonian(full_basis, terms)
    @test P' * P ≈ Matrix{ComplexF64}(I, 4, 4)
    @test block_H ≈ P' * Matrix(full_H) * P
    @test block_diag_hamiltonian(
        blocks,
        terms,
        Any[],
        SpinBasis1D,
        (2,),
        ComplexF64;
        basis_kwargs=Dict(:pauli => false),
        get_proj=false,
    ) ≈ block_H

    blocks_operator = BlockOps(
        blocks,
        terms,
        Any[],
        SpinBasis1D,
        (2,),
        ComplexF64;
        basis_kwargs=Dict(:pauli => false),
    )
    @test length(blocks_operator.basis_dict) == 3
    @test isempty(blocks_operator.H_dict)
    compute_all_blocks!(blocks_operator)
    @test length(blocks_operator.H_dict) == 3
    @test length(blocks_operator.P_dict) == 3
    psi = normalize(ComplexF64[1, 2, 3, 4])
    times = [0.0, 0.2, 1.0]
    @test evolve(blocks_operator, psi, 0.0, times) ≈
        evolve(full_H, psi, 0.0, times) atol=5e-15
    @test block_expm(blocks_operator, psi) ≈
        exp((-im) .* Matrix(full_H)) * psi atol=5e-15
    grid = block_expm(blocks_operator, psi; start=0.0, stop=1.0, num=3)
    @test grid[:, 1] ≈ psi atol=3e-16
    @test grid[:, end] ≈ exp((-im) .* Matrix(full_H)) * psi atol=5e-15
end

@testset "Entanglement and time-dependent measurements" begin
    basis = SpinBasis1D(4)
    ghz = zeros(ComplexF64, length(basis))
    ghz[state_index(basis, 0)] = inv(sqrt(2))
    ghz[state_index(basis, 15)] = inv(sqrt(2))
    entropy = ent_entropy(basis, ghz; return_rdm=:both)
    @test entropy["Sent_A"] ≈ log(2) / 2 atol=2e-16
    @test entropy["Sent_B"] ≈ log(2) / 2 atol=2e-16
    @test tr(entropy["rdm_A"]) ≈ 1.0 atol=3e-16
    @test partial_trace(basis, ghz; return_rdm=:A) == entropy["rdm_A"]
    deprecated = ent_entropy(ghz, basis; DM=:both)
    @test deprecated["Sent"] == deprecated["Sent_A"]
    @test deprecated["DM_chain_subsys"] == deprecated["rdm_A"]

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
    )
    @test observations["A"] ≈ ones(3) atol=1e-15
    @test observations["B"] ≈
        [2.0, 1.99255385, 1.89288098] atol=6e-9
    @test size(observations["psi_t"]) == (4, 3)
    dynamic = obs_vs_time(
        observations["psi_t"],
        times,
        Dict("scaled_identity" => time -> time * Matrix{Float64}(I, 4, 4));
        enforce_pure=true,
    )
    @test real.(dynamic["scaled_identity"]) ≈ times atol=2e-15
end

@testset "Diagonal ensemble measurements" begin
    V2 = [1.0 1.0; 1.0 -1.0] ./ sqrt(2)
    E2 = [-1.0, 1.0]
    psi = [1.0, 0.0]
    observable = Diagonal([1.0, -1.0])
    result = diag_ensemble(
        2,
        psi,
        E2,
        V2;
        density=false,
        rho_d=true,
        Obs=observable,
        delta_t_Obs=true,
        delta_q_Obs=true,
        Sd_Renyi=true,
    )
    @test result["rho_d"] ≈ [0.5, 0.5] atol=3e-16
    @test result["Obs_pure"] ≈ 0.0 atol=3e-16
    @test result["delta_t_Obs_pure"] ≈ inv(sqrt(2)) atol=4e-16
    @test result["delta_q_Obs_pure"] ≈ inv(sqrt(2)) atol=4e-16
    @test result["Sd_pure"] ≈ log(2) atol=3e-16

    mixed = diag_ensemble(
        2,
        Diagonal([0.75, 0.25]),
        E2,
        Matrix(I, 2, 2);
        density=false,
        rho_d=true,
        Obs=observable,
        Sd_Renyi=true,
    )
    @test mixed["rho_d"] ≈ [0.75, 0.25]
    @test mixed["Obs_DM"] ≈ 0.5
    @test mixed["Sd_DM"] ≈ 0.5623351446188083 atol=3e-16
    @test_throws ArgumentError diag_ensemble(2, psi, [0.0, 0.0], V2)
end

@testset "ExpOp matrix exponential protocol" begin
    O = [0.0 1.0; -1.0 0.0]
    vector = [1.0, 0.0]
    observable = Diagonal([2.0, 3.0])
    expO = ExpOp(O; a=0.5)
    @test isexp_op(expO)
    @test !isexp_op(O)
    @test expO.Ns == 2
    @test expO.get_shape == (2, 2)
    @test expO.ndim == 2
    @test expO.O == O
    @test expO.a == 0.5
    @test expO.grid === nothing
    @test expO.step === nothing
    @test !expO.iterate
    @test get_mat(expO) ≈ [
        0.8775825618903728 0.479425538604203
        -0.47942553860420306 0.8775825618903728
    ] atol=3e-16
    @test apply(expO, vector) ≈
        [0.8775825618903728, -0.479425538604203] atol=3e-16
    @test right_apply(expO, vector) ≈
        [0.8775825618903728, 0.479425538604203] atol=3e-16
    @test sandwich(expO, observable) ≈ [
        2.2298488470659295 0.4207354924039482
        0.4207354924039482 2.7701511529340697
    ] atol=2e-15
    @test expO * vector == apply(expO, vector)
    @test get_mat(expO.H) ≈ get_mat(expO)' atol=3e-16
    @test get_mat(expO.T) ≈ transpose(get_mat(expO)) atol=3e-16
    @test get_mat(conj(expO)) ≈ conj(get_mat(expO)) atol=3e-16
    copied = copy(expO)
    @test copied !== expO
    @test get_mat(copied) == get_mat(expO)

    set_a!(copied, 0.25)
    @test copied.a == 0.25
    set_grid!(copied, 0.0, 1.0; num=3)
    @test copied.grid == [0.0, 0.5, 1.0]
    @test copied.step == 0.5
    @test size(apply(copied, vector)) == (2, 3)
    set_iterate!(copied, true)
    @test length(collect(apply(copied, vector))) == 3
    unset_grid!(copied)
    @test copied.grid === nothing
    @test !copied.iterate
    @test_throws ArgumentError set_iterate!(copied, true)
end

@testset "SpinBasis1D" begin
    basis = SpinBasis1D(4; nup=2, pauli=false)
    @test length(basis) == 6
    @test basis.N == basis.L == 4
    @test basis.Ns == 6
    @test basis.sps == 2
    @test basis.dtype === UInt64
    @test basis.states == states(basis)
    @test basis.blocks[:nup] == 2
    @test "z" in basis.operators
    @test isempty(basis.noncommuting_bits)
    @test !isempty(basis.description)
    @test Set(states(basis)) == Set(UInt64[3, 5, 6, 9, 10, 12])
    @test all(state_at(basis, state_index(basis, s)) == s for s in states(basis))
    @test int_to_state(basis, 3) == "|0 0 1 1>"
    @test int_to_state(basis, 3; bracket_notation=false) == "0011"
    @test state_to_int(basis, "|0 0 1 1>") == 3
    @test state_to_int(basis, "0011") == 3
    @test operator_matrix(basis, "z", [(1.0, 1)]) ==
        Matrix(Hamiltonian(basis, [OperatorTerm("z", [(1.0, 1)])]))
    spin_operator = zeros(ComplexF64, basis.Ns, basis.Ns)
    @test inplace_op!(spin_operator, basis, "z", [(1.0, 1)]) === spin_operator
    @test spin_operator == operator_matrix(basis, "z", [(1.0, 1)])
    @test SpinBasisGeneral !== SpinBasis1D
    @test states(SpinBasisGeneral(4; Nup=2, pauli=false)) == states(basis)
    projector = projection_matrix(basis)
    @test size(projector) == (16, 6)
    @test projector' * projector == Matrix(I, 6, 6)
    vector = collect(1.0:6.0)
    @test get_vec(basis, vector) == projector * vector
    @test project_from(basis, vector) == projector * vector
    static, dynamic = expanded_form(basis, [:static], [:dynamic])
    @test static == [:static]
    @test dynamic == [:dynamic]
    @test_throws ArgumentError state_index(basis, 0)
    @test_throws ArgumentError SpinBasis1D(4; nup=5)
end

@testset "Boson and fermion bases" begin
    bosons = BosonBasis1D(3; Nb=2)
    @test bosons.Ns == 6
    @test bosons.sps == 3
    @test all(sum(row) == 2 for row in eachrow(bosons.occupations))
    @test state_to_int(bosons, int_to_state(bosons, 10)) == 10
    @test projection_matrix(bosons)' * projection_matrix(bosons) ≈
        Matrix(I, 6, 6)
    number_site_1 = operator_matrix(bosons, "n", [(1.0, 1)])
    @test diag(number_site_1) == bosons.occupations[:, 1]
    boson_state = zeros(ComplexF64, bosons.Ns)
    boson_state[1] = 1
    @test tr(partial_trace(bosons, boson_state)) ≈ 1

    fermions = SpinlessFermionBasis1D(4; Nf=2)
    @test fermions.Ns == 6
    @test all(sum(row) == 2 for row in eachrow(fermions.occupations))
    hopping = operator_matrix(
        fermions,
        "+-",
        [(1.0, 1, 2), (1.0, 2, 1)],
    )
    @test ishermitian(hopping)

    unrestricted = SpinlessFermionBasis1D(2)
    creation_at_2 = operator_matrix(unrestricted, "+", [(1.0, 2)])
    @test creation_at_2[state_index(unrestricted, 3), state_index(unrestricted, 1)] == -1

    spinful = SpinfulFermionBasis1D(2; Nf=(1, 1))
    @test spinful.Ns == 4
    @test all(
        count(digit -> digit & 1 == 1, row) == 1 &&
        count(digit -> digit & 2 == 2, row) == 1
        for row in eachrow(spinful.occupations)
    )
    @test BosonBasisGeneral !== BosonBasis1D
    @test SpinlessFermionBasisGeneral !== SpinlessFermionBasis1D
    @test SpinfulFermionBasisGeneral !== SpinfulFermionBasis1D
    @test states(BosonBasisGeneral(3; Nb=2)) == states(bosons)
    @test states(SpinlessFermionBasisGeneral(4; Nf=2)) == states(fermions)
    @test states(SpinfulFermionBasisGeneral(2; Nf=(1, 1))) == states(spinful)
end

@testset "Tensor and photon bases" begin
    spin = SpinBasis1D(1; pauli=false)
    boson = BosonBasis1D(1; sps=3)
    tensor = TensorBasis(spin, boson)
    @test tensor.Ns == 6
    @test tensor.N == (1, 1)
    @test tensor.sps == (2, 3)
    @test tensor.basis_left === spin
    @test tensor.basis_right === boson
    @test projection_matrix(tensor) ≈ Matrix(I, 6, 6)
    @test state_index(tensor, 2, 2) == 5
    combined = operator_matrix(tensor, "z|n", [(1.0, 1, 1)])
    @test combined ≈ kron(
        operator_matrix(spin, "z", [(1.0, 1)]),
        operator_matrix(boson, "n", [(1.0, 1)]),
    )
    entangled = zeros(ComplexF64, 6)
    entangled[state_index(tensor, 1, 1)] = inv(sqrt(2))
    entangled[state_index(tensor, 2, 2)] = inv(sqrt(2))
    @test ent_entropy(tensor, entangled)["Sent_A"] ≈ log(2) atol=3e-16

    photons = PhotonBasis(SpinBasis1D, 1; Nph=2, pauli=false)
    @test photons.Ns == 6
    @test photons.particle_Ns == 2
    @test photons.photon_Ns == 3
    @test photons.photon_sps == 3
    @test photons.basis_left === photons.basis_particle
    @test photons.basis_right === photons.basis_photon
    @test operator_matrix(photons, "z|n", [(1.0, 1, 1)]) ≈ combined
end

@testset "User-defined basis" begin
    operators = Dict(
        "n" => [0.0 0.0; 0.0 1.0],
        "+" => [0.0 0.0; 1.0 0.0],
        "x" => ((state, site) -> (xor(state, UInt64(1) << (site - 1)), 1.0)),
    )
    basis = UserBasis(UInt64, 2, operators)
    @test basis.Ns == 4
    @test basis.sps == 2
    @test Set(basis.states) == Set(UInt64[0, 1, 2, 3])
    @test diag(operator_matrix(basis, "n", [(1.0, 1)])) ==
        ComplexF64[0, 1, 0, 1]
    @test operator_matrix(basis, "x", [(1.0, 1)]) ≈
        ComplexF64[0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]

    even = UserBasis(
        UInt64,
        2,
        operators;
        pre_check_state=state -> iseven(count_ones(state)),
        allowed_ops=("n", "x"),
    )
    @test even.Ns == 2
    @test even.states == UInt64[0, 3]
    state = ComplexF64[inv(sqrt(2)), inv(sqrt(2))]
    @test ent_entropy(even, state; sub_sys_A=[1])["Sent_A"] ≈ log(2) atol=3e-16
    @test projection_matrix(even)' * projection_matrix(even) ≈ Matrix(I, 2, 2)
end

@testset "XXZ spectrum" begin
    basis = SpinBasis1D(4; nup=2, pauli=false)
    jxy = sqrt(2.0)
    terms = [
        OperatorTerm("+-", [(jxy / 2, i, i + 1) for i in 1:3]),
        OperatorTerm("-+", [(jxy / 2, i, i + 1) for i in 1:3]),
        OperatorTerm("zz", [(1.0, i, i + 1) for i in 1:3]),
        OperatorTerm("z", [(inv(sqrt(3.0)), i) for i in 1:4]),
    ]
    H = Hamiltonian(basis, terms)
    @test ishermitian(H)
    @test eigvals(H) ≈ [
        -2.06671263224485,
        -1.116025403784439,
        -0.25,
        0.13427005520587385,
        0.6160254037844387,
        1.1824425770389764,
    ] atol=1e-12 rtol=1e-12
    @test tr(Matrix(H)) ≈ -1.5 atol=1e-12
    @test norm(Matrix(H)) ≈ 2.715695122800054 atol=1e-12
end

@testset "Hamiltonian general constructor and dynamics" begin
    basis = SpinBasis1D(1; pauli=false)
    X = ComplexF64[0 0.5; 0.5 0]
    drive = (time, frequency) -> cos(frequency * time)
    H = Hamiltonian(
        Any[Any["z", [(1.0, 1)]]],
        Any[Any[X, drive, (2.0,)]];
        basis,
        dtype=ComplexF64,
    )
    Z = ComplexF64[-0.5 0; 0 0.5]
    @test H.Ns == 2
    @test length(H.dynamic) == 1
    @test toarray(H; time=0.0) ≈ Z + X
    @test toarray(H; time=π / 4) ≈ Z atol=3e-16
    @test apply(H, ComplexF64[1, 0]; time=π / 4) ≈ Z * ComplexF64[1, 0]
    @test eigh(H; time=0.0)[1] ≈ eigvals(Hermitian(Z + X))

    constant = Hamiltonian(
        Any[Z],
        Any[Any[X, (time,) -> 1.0, ()]];
        basis,
        dtype=ComplexF64,
    )
    psi = ComplexF64[1, 0]
    evolved = evolve(constant, psi, 0.0, [0.0, 0.2]; max_step=0.001)
    @test evolved[:, end] ≈ exp((-0.2im) .* (Z + X)) * psi atol=2e-12
end

@testset "Hamiltonian linear algebra protocol" begin
    basis = SpinBasis1D(2; pauli=false)
    H = Hamiltonian(
        basis,
        [
            OperatorTerm("+-", [(0.5, 1, 2)]),
            OperatorTerm("-+", [(0.5, 1, 2)]),
            OperatorTerm("z", [(0.3, 1)]),
        ],
    )
    matrix = Matrix(H)
    @test ishamiltonian(H)
    @test !ishamiltonian(matrix)
    @test H.Ns == 4
    @test H.shape == size(matrix) == H.get_shape
    @test H.ndim == 2
    @test H.dtype == eltype(matrix)
    @test H.basis === basis
    @test H.static == matrix
    @test isempty(H.dynamic)
    @test H.is_dense
    @test H.nbytes >= sizeof(matrix)
    @test H.H.data == matrix'
    @test H.T.data == transpose(matrix)

    @test as_dense_format(H) === H
    @test as_dense_format(H; copy=true) !== H
    @test Matrix(as_sparse_format(H)) == matrix
    @test aslinearoperator(H) === H
    @test check_is_dense(H)
    @test astype(H, ComplexF64).dtype == ComplexF64
    @test Matrix(conj(H)) == conj(matrix)
    @test Matrix(transpose(H)) == transpose(matrix)
    @test Matrix(adjoint(H)) == matrix'
    @test copy(H) !== H
    @test diagonal(H) == diag(matrix)

    vector = ComplexF64[1, 2, -1, 0.5]
    @test apply(H, vector) == matrix * vector
    out = zeros(ComplexF64, 4)
    @test apply(H, vector; out, a=2) === out
    @test out == 2matrix * vector
    @test right_apply(H, vector) == vec(transpose(vector) * matrix)
    values, vectors = eigh(H)
    @test matrix * vectors ≈ vectors * Diagonal(values) atol=1e-14
    sparse_values, sparse_vectors = eigsh(H; k=2, which=:SA)
    @test sparse_values ≈ values[1:2] atol=2e-13
    @test size(sparse_vectors) == (4, 2)
    @test eigsh(H; k=2, which=:SA, return_eigenvectors=false) ≈
        values[1:2] atol=2e-13

    # This nonsingular symmetric matrix has a zero first LDLt pivot.
    # Shift-invert therefore needs a pivoted factorization.
    pivot_basis = SpinBasis1D(2)
    pivot_H = Hamiltonian(
        pivot_basis,
        [OperatorTerm("x", [(1.0, 1)])];
        static_fmt=:csc,
    )
    pivot_values, pivot_vectors = eigsh(
        pivot_H;
        k=1,
        sigma=0.0,
        which=:LM,
        v0=fill(0.5, length(pivot_basis)),
        tol=1e-12,
    )
    @test abs(pivot_values[1]) ≈ 1.0 atol=2e-13
    @test norm(
        Matrix(pivot_H) * pivot_vectors -
        pivot_vectors * Diagonal(pivot_values),
    ) < 2e-12
    pivot_values_only = eigsh(
        pivot_H;
        k=1,
        sigma=0.0,
        which=:LM,
        return_eigenvectors=false,
        v0=fill(0.5, length(pivot_basis)),
        tol=1e-12,
    )
    @test abs(pivot_values_only[1]) ≈ 1.0 atol=2e-13

    normalized = vector / norm(vector)
    @test expt_value(H, normalized) ≈ dot(normalized, matrix * normalized)
    @test matrix_ele(H, normalized, normalized) ≈ expt_value(H, normalized)
    @test quant_fluct(H, normalized) ≈
        dot(normalized, matrix^2 * normalized) - expt_value(H, normalized)^2
    identity_projector = Matrix{Float64}(I, 4, 4)
    @test Matrix(project_to(H, identity_projector)) == matrix
    rotation = ExpOp(H; a=-0.2im)
    @test Matrix(rotate_by(H, rotation)) ≈
        get_mat(rotation)' * matrix * get_mat(rotation) atol=2e-14

    times = [0.0, 0.25, 1.0]
    evolved = evolve(H, normalized, 0.0, times)
    @test evolved[:, 1] ≈ normalized atol=2e-15
    @test all(isapprox(norm(column), 1.0; atol=2e-14) for column in eachcol(evolved))
    @test toarray(H) == matrix
    @test todense(H) == matrix
    @test Matrix(tocsc(H)) == matrix
    @test Matrix(tocsr(H)) == matrix
    @test tr(H) == tr(matrix)
    @test update_matrix_formats!(H, :csc, Dict()) === H
    @test H.data isa SparseMatrixCSC
    @test !H.is_dense
    @test update_matrix_formats!(H, :dense, Dict()) === H
    @test H.data isa Matrix
    @test update_matrix_formats!(H, :csr) === H
    @test H.data isa SparseMatrixCSR
    for storage in (H.data, DIAMatrix(sparse(H.data)))
        destination = ComplexF64[0.5, -0.25, 0.75, -1.0]
        original_destination = copy(destination)
        @test mul!(destination, storage, vector, 1.25, -0.5) ===
            destination
        @test destination ≈
            1.25 .* (storage * vector) .-
            0.5 .* original_destination atol=3e-16
    end
end

@testset "QuantumOperator native archive" begin
    basis = SpinBasis1D(2)
    operator = QuantumOperator(
        basis,
        Dict(
            "z" => [OperatorTerm("z", [(1.0, 1)])],
            "x" => [
                OperatorTerm("+", [(0.5, 2)]),
                OperatorTerm("-", [(0.5, 2)]),
            ],
        ),
    )
    mktempdir() do directory
        archive = joinpath(directory, "operator.zip")
        @test save_zip(archive, operator) == archive
        restored = load_zip(archive)
        @test restored.basis == operator.basis
        @test restored.components == operator.components

        basisless = joinpath(directory, "operator-no-basis.zip")
        save_zip(basisless, operator, false)
        inferred = load_zip(basisless)
        @test inferred.basis.L == 2
        @test inferred.components == operator.components
    end
end

@testset "QuantumOperator Python-compatible archive" begin
    basis = SpinBasis1D(2)
    dense_component = ComplexF64[
        1 2im 0 0
        -2im 3 0 0
        0 0 4 0.5
        0 0 0.5 5
    ]
    sparse_component = sparse(ComplexF64[
        0 0.25 0 0
        0.25 0 0 0
        0 0 0 -0.75im
        0 0 0.75im 0
    ])
    operator = QuantumOperator(
        basis,
        Dict(
            "dense" => dense_component,
            "sparse" => sparse_component,
        );
        matrix_formats=Dict("dense" => :dense, "sparse" => :csc),
    )
    mktempdir() do directory
        archive = joinpath(directory, "python-compatible.zip")
        @test save_zip(
            archive,
            operator;
            save_basis=false,
            format=:python,
        ) == archive
        restored = load_zip(archive)
        @test restored.basis == basis
        @test restored.components["dense"] == dense_component
        @test restored.components["sparse"] isa SparseMatrixCSC
        @test restored.components["sparse"] == sparse_component
        @test_throws ArgumentError save_zip(
            joinpath(directory, "unsafe-basis.zip"),
            operator;
            format=:python,
        )
    end
end

@testset "QuantumLinearOperator protocol" begin
    basis = SpinBasis1D(3; nup=1, pauli=false)
    terms = [
        OperatorTerm("+-", [(0.7, 1, 2), (0.4, 2, 3)]),
        OperatorTerm("-+", [(0.7, 1, 2), (0.4, 2, 3)]),
        OperatorTerm("z", [(0.2, 1), (-0.3, 3)]),
    ]
    operator = QuantumLinearOperator(basis, terms; diagonal=[0.1, -0.2, 0.3])
    matrix = Matrix(Hamiltonian(basis, terms)) + Diagonal([0.1, -0.2, 0.3])
    @test isquantum_LinearOperator(operator)
    @test !isquantum_LinearOperator(matrix)
    @test operator.Ns == 3
    @test operator.shape == (3, 3) == operator.get_shape
    @test operator.ndim == 2
    @test operator.dtype == Float64
    @test operator.basis === basis
    @test operator.static_list == terms
    @test operator.diagonal == [0.1, -0.2, 0.3]
    @test Matrix(operator) == matrix
    @test Matrix(operator.H) == matrix'
    @test Matrix(operator.T) == transpose(matrix)

    vector = normalize(ComplexF64[1, 2im, -0.5])
    @test operator * vector ≈ matrix * vector atol=3e-16
    destination = ComplexF64[0.25, -0.5im, 0.75]
    original_destination = copy(destination)
    @test mul!(destination, operator, vector, 1.25, -0.5) === destination
    @test destination ≈
        1.25 .* (matrix * vector) .- 0.5 .* original_destination atol=3e-16
    @test apply(operator, vector) ≈ matrix * vector atol=3e-16
    @test right_apply(operator, vector) == vec(transpose(vector) * matrix)
    @test Matrix(conj(operator)) == conj(matrix)
    @test Matrix(transpose(operator)) == transpose(matrix)
    @test Matrix(adjoint(operator)) == matrix'
    @test copy(operator) !== operator
    values, vectors = eigsh(operator; k=2, which=:SA)
    @test matrix * vectors ≈ vectors * Diagonal(values) atol=2e-14
    expectation = dot(vector, matrix * vector)
    @test expt_value(operator, vector) ≈ expectation
    @test matrix_ele(operator, vector, vector) ≈ expectation
    @test quant_fluct(operator, vector) ≈
        dot(vector, matrix^2 * vector) - expectation^2 atol=2e-15
    set_diagonal!(operator, zeros(3))
    @test operator.diagonal == zeros(3)
    @test Matrix(operator) == Matrix(Hamiltonian(basis, terms))
end

@testset "QuantumOperator parameter protocol" begin
    basis = SpinBasis1D(2; pauli=false)
    x_terms = [
        OperatorTerm("+-", [(1.0, 1, 2)]),
        OperatorTerm("-+", [(1.0, 1, 2)]),
    ]
    z_terms = [OperatorTerm("z", [(1.0, 1)])]
    operator = QuantumOperator(basis, Dict(:x => x_terms, :z => z_terms))
    pars = Dict(:x => 0.7, :z => 0.3)
    expected = 0.7Matrix(Hamiltonian(basis, x_terms)) +
        0.3Matrix(Hamiltonian(basis, z_terms))

    @test isquantum_operator(operator)
    @test !isquantum_operator(expected)
    @test operator.Ns == 4
    @test operator.shape == (4, 4) == operator.get_shape
    @test operator.ndim == 2
    @test operator.dtype == Float64
    @test operator.basis === basis
    @test operator.is_dense
    @test get_operators(operator, :x) == Matrix(Hamiltonian(basis, x_terms))
    @test toarray(operator; pars) == expected
    @test (@inferred toarray(operator; pars)) == expected
    @test todense(operator; pars) == expected
    @test Matrix(tocsc(operator; pars)) == expected
    @test Matrix(tocsr(operator; pars)) == expected
    @test diagonal(operator; pars) == diag(expected)
    @test tr(operator; pars) == tr(expected)
    @test toarray(operator.H; pars) == expected'
    @test toarray(operator.T; pars) == transpose(expected)

    vector = normalize(ComplexF64[1, 2im, -1, 0.5])
    @test apply(operator, vector; pars) == expected * vector
    @test right_apply(operator, vector; pars) ==
        vec(transpose(vector) * expected)
    @test toarray(conj(operator); pars) == conj(expected)
    @test toarray(transpose(operator); pars) == transpose(expected)
    @test toarray(adjoint(operator); pars) == expected'
    @test copy(operator) !== operator
    @test astype(operator, ComplexF64).dtype == ComplexF64
    values, vectors = eigh(operator; pars)
    @test expected * vectors ≈ vectors * Diagonal(values) atol=2e-14
    @test eigvals(operator; pars) ≈ values atol=2e-14
    selected, selected_vectors = eigsh(operator; pars, k=2, which=:SA)
    @test selected ≈ values[1:2] atol=2e-13
    @test size(selected_vectors) == (4, 2)

    expectation = dot(vector, expected * vector)
    @test expt_value(operator, vector; pars) ≈ expectation
    @test matrix_ele(operator, vector, vector; pars) ≈ expectation
    @test quant_fluct(operator, vector; pars) ≈
        dot(vector, expected^2 * vector) - expectation^2 atol=2e-15
    @test Matrix(tohamiltonian(operator; pars)) == expected
    @test Matrix(aslinearoperator(operator; pars)) == expected
    @test update_matrix_formats!(operator, Dict(:x => :csc)) === operator
    @test operator.components[:x] isa SparseMatrixCSC
    @test operator.is_dense
    @test update_matrix_formats!(operator, Dict(:z => :csc)) === operator
    @test !operator.is_dense
    @test update_matrix_formats!(operator, Dict(:x => :csr)) === operator
    @test operator.components[:x] isa SparseMatrixCSR
end

include("paper_workflows.jl")
include("completeness_gaps.jl")
include("basis_algorithm_regressions.jl")
include("operators_algorithm_regressions.jl")
include("tools_algorithm_regressions.jl")
include("semantic_parity_regressions.jl")
