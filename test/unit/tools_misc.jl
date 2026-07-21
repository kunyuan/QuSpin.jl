using LinearAlgebra

@testset "tools.misc properties" begin
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

    for N in (1, 31, 32, 63, 64, 100, 255)
        values = BigInt[0, 1, (BigInt(1) << (N - 1)), (BigInt(1) << N) - 1]
        T = get_basis_type(N, nothing, 2)
        represented = T.(values)
        @test array_to_ints(ints_to_array(represented, N), T) == represented
    end

    @test kl_div([0.25, 0.75], [0.5, 0.5]) ≈
        0.13081203594113697 atol=2e-16
    @test kl_div([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]) ≈
        0.4564348191467835 atol=2e-16
    @test kl_div([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) ≈ 0.0 atol=1e-16

    @test mean_level_spacing([0.0, 1.0, 3.0, 6.0]) ≈
        0.5833333333333333 atol=1e-16
    @test mean_level_spacing([0.0, 1.0, 2.0, 4.0, 8.0]) ≈
        0.6666666666666666 atol=1e-16
    @test isnan(mean_level_spacing([0.0, 1.0, 1.0, 2.0]; verbose=false))

    A = [1.0 2.0; 3.0 4.0]
    v = [2.0, -1.0]
    @test matvec(A, v; a=2.0) == [0.0, 4.0]
    @test get_matvec_function(A)(A, v; a=2.0) == [0.0, 4.0]
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

    basis = SpinBasis1D(2; pauli=false)
    H = Hamiltonian(basis, [OperatorTerm("z", [(1.0, 1)])])
    @test project_op(H, basis)["Proj_Obs"] == ComplexF64.(Matrix(H))
    @test get_matvec_function(H)(H, ones(length(basis))) ==
        H * ones(length(basis))
end
