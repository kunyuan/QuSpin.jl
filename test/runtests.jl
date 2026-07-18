using Test
using LinearAlgebra
using QuSpin

@testset "SpinBasis1D" begin
    basis = SpinBasis1D(4; nup=2, pauli=false)
    @test length(basis) == 6
    @test Set(states(basis)) == Set(UInt64[3, 5, 6, 9, 10, 12])
    @test all(state_at(basis, state_index(basis, s)) == s for s in states(basis))
    @test_throws ArgumentError state_index(basis, 0)
    @test_throws ArgumentError SpinBasis1D(4; nup=5)
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
