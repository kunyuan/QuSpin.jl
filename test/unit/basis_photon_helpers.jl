@testset "photon helper properties" begin
    # Values captured from QuSpin 1.0.1 on the fixed source snapshot.
    @test coherent_state(0.5, 5) ≈ [
        0.8824969025845955,
        0.4412484512922977,
        0.15600488604842286,
        0.045034731477476914,
        0.011258682869369227,
    ] atol=2e-17 rtol=2e-15
    @test coherent_state(0.5 + 0.25im, 5; dtype=ComplexF64) ≈ [
        0.8553453273074225 + 0.0im,
        0.42767266365371126 + 0.21383633182685563im,
        0.11340384022411978 + 0.15120512029882638im,
        0.010912289613422155 + 0.060017592873821815im,
        -0.004774126705872185 + 0.016368434420133214im,
    ] atol=3e-16 rtol=2e-15

    state = coherent_state(0.3 + 0.2im, 80; dtype=ComplexF64)
    @test sum(abs2, state) ≈ 1.0 atol=2e-15
    @test all(
        isapprox(
            state[n + 1],
            state[n] * (0.3 + 0.2im) / sqrt(n);
            atol=2e-16,
        )
        for n in 1:79
    )

    @test coherent_state(0.0, 5) == [1.0, 0.0, 0.0, 0.0, 0.0]
    @test photon_Hspace_dim(4, nothing, 3) == 64
    @test photon_Hspace_dim(4, 0, nothing) == 1
    @test photon_Hspace_dim(4, 1, nothing) == 5
    @test photon_Hspace_dim(4, 2, nothing) == 11
    @test photon_Hspace_dim(4, 4, nothing) == 16
    @test photon_Hspace_dim(8, 4, nothing) == 163

    for N in 0:12
        @test photon_Hspace_dim(N, N, nothing) == 2^N
        @test photon_Hspace_dim(N, nothing, 0) == 2^N
    end
end

@testset "truncated harmonic oscillator basis" begin
    basis = HOBasis(4)
    @test isbasis(basis)
    @test basis.Np == 4
    @test basis.Ns == 5
    number = operator_matrix(basis, "n", [(1.0, 1)]; sparse=false)
    raising = operator_matrix(basis, "+", [(1.0, 1)]; sparse=false)
    @test number ≈ Diagonal(0.0:4.0) atol=2e-16
    @test raising * raising' ≈ Diagonal(0.0:4.0) atol=4e-15
end
