using Test
using LinearAlgebra
using Random
using SparseArrays
using QuSpin

Random.seed!(0x51a7)

function reference_discrete_csc(basis, specifications, ::Type{T}) where {T}
    rows = Int[]
    columns = Int[]
    values = T[]
    for (opstring, couplings) in specifications
        term_rows, term_columns, term_values =
            QuSpin.Basis._discrete_operator_triplets(
                basis,
                opstring,
                couplings,
            )
        append!(rows, term_rows)
        append!(columns, term_columns)
        append!(values, T.(term_values))
    end
    return sparse(rows, columns, values, length(basis), length(basis))
end

function reference_user_csc(basis, specifications)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    for (opstring, couplings) in specifications
        term_rows, term_columns, term_values =
            QuSpin.Basis._user_operator_triplets(
                basis,
                opstring,
                couplings,
            )
        append!(rows, term_rows)
        append!(columns, term_columns)
        append!(values, term_values)
    end
    return sparse(rows, columns, values, length(basis), length(basis))
end

@testset "generic transition-to-CSC assembly" begin
    cases = [
        (
            BosonBasis1D(4; Nb=3, sps=3),
            [
                ("+-", [(0.7, 1, 2), (-0.2, 3, 4)]),
                ("n", [(0.3, site) for site in 1:4]),
            ],
            Float64,
        ),
        (
            SpinBasis1D(4; S=1, Nup=4),
            [
                ("+-", [(0.4, 1, 2), (-0.6, 3, 4)]),
                ("zz", [(0.2, 1, 3), (-0.1, 2, 4)]),
                ("y", [(0.3, 2)]),
            ],
            ComplexF64,
        ),
        (
            SpinlessFermionBasis1D(6; Nf=3),
            [
                ("+-", [(0.8, 1, 2), (-0.4, 4, 6)]),
                ("n", [(0.25, site) for site in 1:6]),
            ],
            Float64,
        ),
        (
            SpinlessFermionBasis1D(4),
            [
                ("x", [(0.3, 1), (-0.2, 4)]),
                ("y", [(0.1, 2)]),
            ],
            ComplexF64,
        ),
        (
            SpinfulFermionBasis1D(4; Nf=(2, 1)),
            [
                ("+-|", [(0.6, 1, 3), (-0.2, 2, 4)]),
                ("|+-", [(0.5, 1, 4)]),
                ("n|n", [(0.7, 2, 3)]),
            ],
            Float64,
        ),
    ]

    for (basis, specifications, T) in cases
        assembled = QuSpin.Basis._discrete_operator_csc(
            basis,
            specifications,
            T,
        )
        reference = reference_discrete_csc(basis, specifications, T)
        @test assembled == reference
        @test all(
            issorted(rowvals(assembled)[nzrange(assembled, column)])
            for column in axes(assembled, 2)
        )
        @test all(!iszero, nonzeros(assembled))

        terms = [OperatorTerm(opstring, couplings)
                 for (opstring, couplings) in specifications]
        @test Matrix(Hamiltonian(
            basis,
            terms;
            static_fmt=:csc,
            check_symm=false,
            check_herm=false,
            check_pcon=false,
        )) ==
            Matrix(reference)
    end

    randomized_cases = [
        (BosonBasis1D(4; Nb=3, sps=3), ["+-", "-+", "n"]),
        (SpinBasis1D(4; S=1, Nup=4), ["+-", "-+", "zz", "x", "y"]),
        (SpinlessFermionBasis1D(5; Nf=2), ["+-", "-+", "nn"]),
        (SpinlessFermionBasis1D(4), ["+", "-", "x", "y", "n"]),
        (SpinfulFermionBasis1D(3), ["+-|", "|+-", "n|n", "x|", "|y"]),
    ]
    for _ in 1:8, (basis, opstrings) in randomized_cases
        specifications = [
            (
                opstring,
                [
                    (
                        randn() + im * randn(),
                        (
                            rand(1:basis.L)
                            for _ in 1:count(!=('|'), opstring)
                        )...,
                    )
                    for _ in 1:3
                ],
            )
            for opstring in opstrings
        ]
        @test QuSpin.Basis._discrete_operator_csc(
            basis,
            specifications,
            ComplexF64,
        ) ≈ reference_discrete_csc(
            basis,
            specifications,
            ComplexF64,
        ) atol=2e-14
    end

    binary_bosons = BosonBasis1D(6; Nb=3, sps=2)
    indexer = QuSpin.Basis._discrete_state_indexer(binary_bosons)
    @test indexer isa QuSpin.Basis._CombinadicDiscreteStateIndexer
    @test [
        QuSpin.Basis._discrete_state_index(indexer, state)
        for state in binary_bosons.encoded_states
    ] == collect(1:length(binary_bosons))

    spinless = SpinlessFermionBasis1D(6; Nf=3)
    transitions = QuSpin.Basis._compile_discrete_transitions(
        spinless,
        (("+-", [(0.8, 1, 2), (-0.4, 4, 6)]),),
        Float64,
    )
    spinless_indexer = QuSpin.Basis._discrete_state_indexer(spinless)
    @test (@inferred QuSpin.Basis._assemble_discrete_csc(
        spinless,
        transitions,
        spinless_indexer,
    )) isa SparseMatrixCSC{Float64,Int}
    @test (@inferred QuSpin.Basis._discrete_operator_csc(
        spinless,
        (("+-", [(0.8, 1, 2), (-0.4, 4, 6)]),),
        Float64,
    )) isa SparseMatrixCSC{Float64,Int}
end

@testset "UserBasis direct transition-to-CSC assembly" begin
    cycle = function (state, site)
        weight = UInt64(3)^(site - 1)
        old = Int((state ÷ weight) % UInt64(3))
        new = mod(old + 1, 3)
        updated = UInt64(
            Int128(state) + Int128(new - old) * Int128(weight),
        )
        return updated, 0.5 - 0.25im
    end
    branching = ComplexF64[
        1.0 0.5im 0.0
        -0.25 1.0 0.75
        0.5 0.0 -1.0im
    ]
    basis = UserBasis(
        UInt64,
        2,
        Dict('a' => branching, 'c' => cycle);
        sps=3,
        states=UInt64[0, 1, 3, 4, 6, 7],
        allowed_ops=('a', 'c'),
    )
    specifications = [
        ("a", [(0.7 - 0.2im, 1), (-0.3, 2)]),
        ("aa", [(0.4 + 0.1im, 1, 1)]),
        ("c", [(1.1, 2)]),
    ]
    expected = reference_user_csc(basis, specifications)
    assembled = @inferred QuSpin.Basis._user_operator_csc(
        basis,
        specifications,
        ComplexF64,
    )
    @test assembled ≈ expected atol=2e-14
    @test all(
        issorted(rowvals(assembled)[nzrange(assembled, column)])
        for column in axes(assembled, 2)
    )
    @test operator_matrix(
        basis,
        "aa",
        [(0.4 + 0.1im, 1, 1)];
        sparse=true,
    ) == reference_user_csc(
        basis,
        [("aa", [(0.4 + 0.1im, 1, 1)])],
    )

    constrained = constraint_states(
        16;
        prefix_allowed=(occupations, site) ->
            site == 1 ||
            occupations[site - 1] + occupations[site] <= 1,
        state_allowed=occupations ->
            occupations[1] + occupations[end] <= 1,
    )
    pxp = UserBasis(
        UInt64,
        16,
        Dict('x' => ComplexF64[0 1; 1 0]);
        states=constrained,
        allowed_ops=('x',),
    )
    pxp_specifications = [
        ("x", [(1.0, site) for site in 1:16]),
    ]
    QuSpin.Basis._user_operator_csc(
        pxp,
        pxp_specifications,
        Float64,
    )
    allocated = @allocated QuSpin.Basis._user_operator_csc(
        pxp,
        pxp_specifications,
        Float64,
    )
    @test allocated < 10_000_000
end

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
