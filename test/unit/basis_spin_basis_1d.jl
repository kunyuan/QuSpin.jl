@testset "SpinBasis1D public contract" begin
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

    for state in states(basis)
        @test state_at(basis, state_index(basis, state)) == state
        formatted = int_to_state(basis, state)
        @test state_to_int(basis, formatted) == state
    end
    @test int_to_state(basis, 5; bracket_notation=false) == "0101"
    @test operator_matrix(basis, "z", [(1.0, 1)]) ==
        Matrix(Hamiltonian(basis, [OperatorTerm("z", [(1.0, 1)])]))
    output = zeros(ComplexF64, basis.Ns, basis.Ns)
    inplace_op!(output, basis, "z", [(1.0, 1)])
    @test output == operator_matrix(basis, "z", [(1.0, 1)])

    projector = projection_matrix(basis)
    @test size(projector) == (16, 6)
    @test projector' * projector == Matrix(I, 6, 6)
    vector = collect(1.0:6.0)
    @test get_vec(basis, vector) == projector * vector
    @test project_from(basis, vector) == projector * vector
    @test project_from(basis, vector; sparse=true) isa SparseVector
    @test project_from(basis, hcat(vector, vector); sparse=true) isa SparseMatrixCSC
    @test project_from(basis, sparse(vector); sparse=false) isa Vector
    momentum = SpinBasis1D(4; nup=2, pauli=false, kblock=0)
    momentum_state = ComplexF64.(1:length(momentum))
    parent_state = project_from(
        momentum,
        momentum_state;
        sparse=false,
        pcon=true,
    )
    @test length(parent_state) == 6
    @test parent_state ≈
        momentum.symmetry.projector * momentum_state atol=3e-14
    @test project_to(
        momentum,
        parent_state;
        sparse=false,
        pcon=true,
    ) ≈ momentum_state atol=3e-14
    static, dynamic = expanded_form(basis, [:static], [:dynamic])
    @test static == [:static]
    @test dynamic == [:dynamic]
end
