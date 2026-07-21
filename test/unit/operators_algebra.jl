@testset "operator algebra properties" begin
    A = [
        1.0 0.2 0.0 0.0
        0.2 2.0 0.3 0.0
        0.0 0.3 3.0 0.4
        0.0 0.0 0.4 4.0
    ]
    B = Diagonal([2.0, -1.0, 0.5, 3.0])
    @test commutator(A, B) ≈ [
        0.0 -0.6 0.0 0.0
        0.6 0.0 0.45 0.0
        0.0 -0.45 0.0 1.0
        0.0 0.0 -1.0 0.0
    ] atol=1e-15
    @test anti_commutator(A, B) ≈ [
        4.0 0.2 0.0 0.0
        0.2 -4.0 -0.15 0.0
        0.0 -0.15 3.0 1.4
        0.0 0.0 1.4 24.0
    ] atol=1e-15

    basis = SpinBasis1D(3; pauli=false)
    Hx = Hamiltonian(
        basis,
        [OperatorTerm("+-", [(1.0, 1, 2)])];
        check_herm=false,
    )
    Hy = Hamiltonian(
        basis,
        [OperatorTerm("-+", [(1.0, 1, 2)])];
        check_herm=false,
    )
    @test commutator(Hx, Hy) == Matrix(Hx) * Matrix(Hy) - Matrix(Hy) * Matrix(Hx)
    @test anti_commutator(Hx, Hy) ==
        Matrix(Hx) * Matrix(Hy) + Matrix(Hy) * Matrix(Hx)
end
