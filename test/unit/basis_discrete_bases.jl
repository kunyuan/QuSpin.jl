@testset "boson and fermion basis properties" begin
    bosons = BosonBasis1D(3; Nb=2)
    @test bosons.N == 3
    @test bosons.Ns == 6
    @test bosons.sps == 3
    @test Set(Int.(bosons.states)) == Set([2, 4, 6, 10, 12, 18])
    @test all(sum(row) == 2 for row in eachrow(bosons.occupations))
    @test bosons.description == "boson lattice basis"
    @test bosons.dtype == UInt64
    @test bosons.operators == ("I", "+", "-", "n", "z")
    @test isempty(bosons.noncommuting_bits)
    @test state_index(bosons, 10) == 4
    @test state_at(bosons, 4) == 10
    @test state_to_int(bosons, int_to_state(bosons, 10)) == 10

    projector = projection_matrix(bosons, ComplexF64)
    @test size(projector) == (27, 6)
    @test projector' * projector ≈ Matrix{ComplexF64}(I, 6, 6)
    state = normalize(ComplexF64[1, 2, 3, 4, 5, 6])
    @test project_from(bosons, state) == projector * state
    @test project_from(bosons, state; sparse=true) isa SparseVector
    @test project_from(bosons, sparse(state); sparse=false) isa Vector
    @test get_vec(bosons, state) == projector * state
    @test tr(partial_trace(bosons, state; sub_sys_A=[1])) ≈ 1 atol=4e-16
    @test ent_entropy(bosons, state; sub_sys_A=[1])["Sent_A"] >= 0
    @test diag(operator_matrix(bosons, "n", [(1.0, 1)])) ==
        ComplexF64.(bosons.occupations[:, 1])

    spinless = SpinlessFermionBasis1D(4; Nf=2)
    @test spinless.Ns == 6
    @test Set(Int.(spinless.states)) == Set([3, 5, 6, 9, 10, 12])
    @test all(sum(row) == 2 for row in eachrow(spinless.occupations))
    @test !isempty(spinless.noncommuting_bits)
    hopping = operator_matrix(
        spinless,
        "+-",
        [(1.0, 1, 2), (1.0, 2, 1)],
    )
    @test ishermitian(hopping)

    unrestricted = SpinlessFermionBasis1D(2)
    creation = operator_matrix(unrestricted, "+", [(1.0, 2)])
    @test creation[state_index(unrestricted, 3), state_index(unrestricted, 1)] == -1

    spinful = SpinfulFermionBasis1D(2; Nf=(1, 1))
    @test spinful.Ns == 4
    @test all(
        count(digit -> digit & 1 == 1, row) == 1 &&
        count(digit -> digit & 2 == 2, row) == 1
        for row in eachrow(spinful.occupations)
    )
end
@testset "higher-spin angular momentum basis" begin
    basis = SpinBasis1D(3; S="3/2", Nup=3, pauli=false)
    @test basis.sps == 4
    @test all(sum(row) == 3 for row in eachrow(basis.occupations))
    unrestricted = SpinBasis1D(1; S="3/2", pauli=false)
    plus = operator_matrix(unrestricted, "+", [(1.0, 1)])
    minus = operator_matrix(unrestricted, "-", [(1.0, 1)])
    z = operator_matrix(unrestricted, "z", [(1.0, 1)])
    @test plus' ≈ minus atol=3e-15
    @test z' == z
    @test plus * minus - minus * plus ≈ 2z atol=4e-14
end

@testset "discrete Hamiltonian CSC composition" begin
    cases = [
        (
            BosonBasis1D(4; Nb=3, sps=3),
            [
                OperatorTerm("+-", [(0.7, 1, 2), (-0.2, 3, 4)]),
                OperatorTerm("n", [(0.3, site) for site in 1:4]),
            ],
        ),
        (
            SpinBasis1D(4; S=1, Nup=4, pauli=false),
            [
                OperatorTerm("+-", [(0.4, 1, 2), (-0.6, 3, 4)]),
                OperatorTerm("zz", [(0.2, 1, 3), (-0.1, 2, 4)]),
            ],
        ),
        (
            SpinlessFermionBasis1D(6; Nf=3),
            [
                OperatorTerm("+-", [(0.8, 1, 2), (-0.4, 4, 6)]),
                OperatorTerm("n", [(0.25, site) for site in 1:6]),
            ],
        ),
        (
            SpinfulFermionBasis1D(4; Nf=(2, 1)),
            [
                OperatorTerm("+-|", [(0.6, 1, 3), (-0.2, 2, 4)]),
                OperatorTerm("|+-", [(0.5, 1, 4)]),
                OperatorTerm("n|n", [(0.7, 2, 3)]),
            ],
        ),
    ]

    for (basis, terms) in cases
        expected = sum(
            sparse(operator_matrix(
                basis,
                term.op,
                term.couplings;
                sparse=true,
            ))
            for term in terms
        )
        assembled = Hamiltonian(
            basis,
            terms;
            static_fmt=:csc,
            check_symm=false,
            check_herm=false,
            check_pcon=false,
        ).data
        @test assembled isa SparseMatrixCSC
        @test assembled == expected
        @test all(
            issorted(rowvals(assembled)[nzrange(assembled, column)])
            for column in axes(assembled, 2)
        )
        @test all(!iszero, nonzeros(assembled))
    end
end

@testset "spinful multiple sectors and Majorana algebra" begin
    basis = SpinfulFermionBasis1D(2; Nf=[(1, 0), (0, 1)])
    @test length(basis) == 4
    unrestricted = SpinfulFermionBasis1D(1)
    x_up = operator_matrix(unrestricted, "x|", [(1.0, 1)])
    y_up = operator_matrix(unrestricted, "y|", [(1.0, 1)])
    @test x_up' ≈ x_up
    @test y_up' ≈ y_up
    @test x_up^2 ≈ I
    @test y_up^2 ≈ I
    @test x_up * y_up + y_up * x_up ≈ zeros(size(x_up)) atol=3e-15
end

@testset "advanced fermionic particle-hole maps" begin
    even = SpinlessFermionBasisGeneral(
        4;
        Nf=2,
        phblock=([-1, -2, -3, -4], 0),
    )
    odd = SpinlessFermionBasisGeneral(
        4;
        Nf=2,
        phblock=([-1, -2, -3, -4], 1),
    )
    parent = SpinlessFermionBasis1D(4; Nf=2)
    @test length(even) + length(odd) == length(parent)
    @test even.symmetry.projector' * even.symmetry.projector ≈
        Matrix{ComplexF64}(I, length(even), length(even)) atol=4e-14
    @test odd.symmetry.projector' * odd.symmetry.projector ≈
        Matrix{ComplexF64}(I, length(odd), length(odd)) atol=4e-14
    @test even.symmetry.projector' * odd.symmetry.projector ≈
        zeros(ComplexF64, length(even), length(odd)) atol=4e-14

    spinful = SpinfulFermionBasisGeneral(
        1;
        Nf=[(0, 0), (1, 1)],
        simple_symm=false,
        phblock=([-1, -2], 0),
    )
    @test length(spinful) == 1
    @test abs.(spinful.symmetry.projector[:, 1]) ≈
        fill(inv(sqrt(2)), 2) atol=4e-14
end

@testset "discrete cross-sector operator application" begin
    source = SpinlessFermionBasisGeneral(4; Nf=1)
    target = SpinlessFermionBasisGeneral(4; Nf=2)
    state = normalize(ComplexF64[1, 2im, -0.4, 0.7])
    shifted = op_shift_sector(
        target,
        source,
        [("+", [3], -0.9)],
        state,
    )
    full = SpinlessFermionBasis1D(4)
    reference =
        projection_matrix(target, ComplexF64; sparse=true)' *
        operator_matrix(full, "+", [(-0.9, 3)]) *
        projection_matrix(source, ComplexF64; sparse=true) *
        state
    @test shifted ≈ reference atol=3e-14

    matrix_elements, bras, kets = op_bra_ket(
        source,
        "+",
        [3],
        -0.9,
        ComplexF64,
        UInt64[0, 1, 2],
    )
    @test matrix_elements == ComplexF64[-0.9, 0.9, 0.9]
    @test bras == UInt64[4, 5, 6]
    @test kets == UInt64[0, 1, 2]
end
