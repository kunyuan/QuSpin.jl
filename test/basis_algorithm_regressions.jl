using Test
using LinearAlgebra
using Random
using SparseArrays
using QuSpin

Random.seed!(0x51a7)

@testset "lazy identity projection" begin
    spin = SpinBasis1D(12; nup=6)
    boson = BosonBasis1D(6; Nb=4, sps=3)
    @test size(spin.symmetry.projector) == (0, 0)
    @test size(boson.symmetry.projector) == (0, 0)

    vector = randn(length(spin))
    vectors = randn(length(spin), 3)
    @test project_from(spin, vector) ≈
        projection_matrix(spin; sparse=true) * vector
    @test project_from(spin, vectors) ≈
        projection_matrix(spin; sparse=true) * vectors

    project_from(spin, vector)
    allocated = @allocated project_from(spin, vector)
    @test allocated < 1_000_000
end

@testset "symmetry trace stays in the parent sector" begin
    spin = SpinBasis1D(8; nup=4, kblock=0)
    spin_state = normalize!(randn(ComplexF64, length(spin)))
    spin_density = spin_state * spin_state'
    spin_projector = projection_matrix(spin, ComplexF64; sparse=true)
    full_spin = SpinBasis1D(8)

    for input in (spin_state, spin_density)
        reduced = partial_trace(
            spin,
            input;
            sub_sys_A=[1, 3, 5],
            return_rdm=:both,
        )
        expanded = input isa AbstractVector ?
            spin_projector * input :
            spin_projector * input * spin_projector'
        reference = partial_trace(
            full_spin,
            expanded;
            sub_sys_A=[1, 3, 5],
            return_rdm=:both,
        )
        @test reduced[1] ≈ reference[1]
        @test reduced[2] ≈ reference[2]
    end

    boson = BosonBasis1D(5; Nb=4, sps=3, kblock=0)
    @test boson.symmetry.parent_occupations[] === nothing
    boson_state = normalize!(randn(ComplexF64, length(boson)))
    boson_projector = projection_matrix(boson, ComplexF64; sparse=true)
    reduced = partial_trace(
        boson,
        boson_state;
        sub_sys_A=[1, 2],
        return_rdm=:both,
    )
    reference = partial_trace(
        BosonBasis1D(5; sps=3),
        boson_projector * boson_state;
        sub_sys_A=[1, 2],
        return_rdm=:both,
    )
    @test reduced[1] ≈ reference[1]
    @test reduced[2] ≈ reference[2]
    @test boson.symmetry.parent_occupations[] !== nothing
end

@testset "Schmidt entropy and grouped mixed trace" begin
    basis = SpinBasis1D(8; nup=4)
    state = normalize!(randn(ComplexF64, length(basis)))
    rho_A = partial_trace(
        basis,
        state;
        sub_sys_A=[1, 2, 3],
        return_rdm=:A,
    )
    probabilities = eigvals(Hermitian(rho_A))
    probabilities = probabilities[probabilities .> 1e-12]
    reference = -sum(probabilities .* log.(probabilities)) / 3
    result = ent_entropy(
        basis,
        state;
        sub_sys_A=[1, 2, 3],
        return_rdm=:both,
        return_rdm_EVs=true,
    )
    @test result["Sent_A"] ≈ reference
    @test result["Sent_B"] ≈ 3reference / 5
    @test result["rdm_A"] ≈ rho_A
    @test length(result["p_A"]) == 8

    mixed = state * state'
    @test partial_trace(
        basis,
        mixed;
        sub_sys_A=[1, 2, 3],
        return_rdm=:A,
    ) ≈ rho_A
end

@testset "factorized tensor projection and static recursion" begin
    left = SpinBasis1D(6; nup=3, kblock=0)
    right = BosonBasis1D(4; Nb=2, sps=3, kblock=0)
    tensor = TensorBasis(left, right)
    vector = randn(ComplexF64, length(tensor))
    vectors = randn(ComplexF64, length(tensor), 2)
    projector = projection_matrix(tensor, ComplexF64; sparse=true)
    @test project_from(tensor, vector) ≈ projector * vector
    @test project_from(tensor, vectors) ≈ projector * vectors

    four_factors = TensorBasis(
        SpinBasis1D(2),
        SpinBasis1D(2),
        SpinBasis1D(2),
        SpinBasis1D(2),
    )
    @test isconcretetype(typeof(four_factors))
    @test length(four_factors) == 256
    inferred_four = @inferred TensorBasis(
        SpinBasis1D(2),
        SpinBasis1D(2),
        SpinBasis1D(2),
        SpinBasis1D(2),
    )
    @test typeof(inferred_four) === typeof(four_factors)
    nested_vector = randn(length(four_factors))
    @test project_from(four_factors, nested_vector) ≈
        projection_matrix(four_factors; sparse=true) * nested_vector
end

@testset "operator hot paths and direct inplace accumulation" begin
    fermions = SpinfulFermionBasis1D(5; Nf=(2, 2))
    couplings = [(0.7, 2, 4), (-0.2, 1, 5)]
    matrix = operator_matrix(fermions, "+|-", couplings)
    out = zeros(ComplexF64, size(matrix))
    @test inplace_op!(out, fermions, "+|-", couplings) === out
    @test out ≈ matrix

    reduced_spin = SpinBasis1D(8; nup=4, kblock=0, pauli=false)
    reduced_couplings = [(0.4, site) for site in 1:8]
    reduced_matrix = operator_matrix(
        reduced_spin,
        "z",
        reduced_couplings,
    )
    reduced_out = zeros(ComplexF64, size(reduced_matrix))
    @test inplace_op!(
        reduced_out,
        reduced_spin,
        "z",
        reduced_couplings,
    ) === reduced_out
    @test reduced_out ≈ reduced_matrix atol=2e-14

    flip = (state, site) -> (
        xor(UInt64(state), UInt64(1) << (site - 1)),
        1.0,
    )
    user = UserBasis(UInt64, 8, Dict('x' => flip))
    user_matrix = operator_matrix(user, "x", [(1.0, 3)])
    user_out = zeros(ComplexF64, size(user_matrix))
    operator_matrix(user, "x", [(1.0, 3)])
    allocated = @allocated operator_matrix(user, "x", [(1.0, 3)])
    @test inplace_op!(user_out, user, "x", [(1.0, 3)]) === user_out
    @test user_out == user_matrix
    @test allocated < 2_000_000

    next_fixed_weight = function (state, counter, N, arguments)
        iszero(state) && return state
        prefix = (state | (state - UInt64(1))) + UInt64(1)
        return prefix | (
            (
                (
                    (prefix & (-prefix)) ÷
                    (state & (-state))
                ) >> 1
            ) - UInt64(1)
        )
    end
    conserved_user = UserBasis(
        UInt64,
        24,
        Dict('x' => flip);
        pcon_dict=Dict(
            :Np => 2,
            :next_state => next_fixed_weight,
            :next_state_args => (),
            :get_Ns_pcon => (N, Np) -> binomial(N, Np),
            :get_s0_pcon => (N, Np) -> (UInt64(1) << Np) - UInt64(1),
        ),
    )
    @test length(conserved_user) == binomial(24, 2)
    @test all(count_ones(state) == 2 for state in states(conserved_user))

    tensor = TensorBasis(SpinBasis1D(3), BosonBasis1D(2; sps=3))
    tensor_couplings = [(0.3, 2)]
    tensor_matrix = operator_matrix(tensor, "|n", tensor_couplings)
    tensor_out = zeros(ComplexF64, size(tensor_matrix))
    @test tensor_matrix ≈ kron(
        0.3 * Matrix{ComplexF64}(I, 8, 8),
        operator_matrix(
            tensor.basis_right,
            "n",
            [(1.0, 2)],
        ),
    )
    @test inplace_op!(tensor_out, tensor, "|n", tensor_couplings) ===
        tensor_out
    @test tensor_out ≈ tensor_matrix
end

@testset "guarded generic symmetry fallback" begin
    dimension = QuSpin.Basis._MAX_DENSE_SYMMETRY_FALLBACK + 1
    projector = SparseMatrixCSC{ComplexF64,Int}(
        spdiagm(0 => ones(ComplexF64, dimension)),
    )
    symmetry = projector + sparse(
        [1],
        [2],
        ComplexF64[0.1],
        dimension,
        dimension,
    )
    @test_throws ArgumentError QuSpin.Basis._intersect_eigenspace(
        projector,
        symmetry,
        1.0 + 0im,
    )
end
