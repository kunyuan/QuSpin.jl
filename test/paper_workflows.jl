@testset "paper workflow: random-field XXZ mid-spectrum states" begin
    # Pal and Huse, Phys. Rev. B 82, 174411 (2010), arXiv:1003.2613.
    # This is the sparse, fixed-magnetization exact-diagonalization kernel used
    # to study the many-body localization transition. The small deterministic
    # instance is a workflow check, not an attempted reproduction of the paper.
    L = 10
    basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
    fields = [-2.3, 1.7, -0.4, 2.8, -1.1, 0.6, -2.0, 1.2, 2.5, -0.9]
    bonds = [(1.0, site, mod1(site + 1, L)) for site in 1:L]
    terms = [
        OperatorTerm("+-", [(0.5, i, j) for (_, i, j) in bonds]),
        OperatorTerm("-+", [(0.5, i, j) for (_, i, j) in bonds]),
        OperatorTerm("zz", bonds),
        OperatorTerm("z", [(fields[site], site) for site in 1:L]),
    ]

    H = Hamiltonian(basis, terms; static_fmt=:csc)
    @test H.data isa SparseMatrixCSC
    @test !H.is_dense
    @test nnz(H.data) < length(basis)^2 ÷ 10

    v0 = normalize(collect(1.0:length(basis)))
    values, vectors = eigsh(
        H;
        k=4,
        sigma=0.0,
        which=:LM,
        v0,
        tol=1e-11,
        maxiter=2_000,
    )
    dense_values = eigvals(Hermitian(Matrix(H)))
    expected = sort(dense_values; by=abs)[1:4]
    @test sort(values; by=abs) ≈ expected atol=2e-9
    @test norm(Matrix(H) * vectors - vectors * Diagonal(values)) < 2e-8
end

@testset "paper workflow: sparse Lanczos XXZ quench" begin
    # The QuSpin paper (SciPost Phys. 2, 003 (2017), arXiv:1610.03042)
    # demonstrates Lanczos time evolution for the Heisenberg chain. Quantum
    # quenches of XXZ chains use this same Hamiltonian-action workflow.
    L = 10
    basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
    bonds = [(1.0, site, mod1(site + 1, L)) for site in 1:L]
    terms = [
        OperatorTerm("+-", [(0.5, i, j) for (_, i, j) in bonds]),
        OperatorTerm("-+", [(0.5, i, j) for (_, i, j) in bonds]),
        OperatorTerm("zz", bonds),
    ]
    H = Hamiltonian(basis, terms; static_fmt=:csc)

    neel_state = sum(UInt64(1) << (site - 1) for site in 1:2:L)
    psi0 = zeros(ComplexF64, length(basis))
    psi0[state_index(basis, neel_state)] = 1
    E, V, Q_T = lanczos_full(H, psi0, 40; full_ortho=true)
    psi_lanczos = expm_lanczos(E, V, Q_T; a=-0.4im)
    psi_exact = exp((-0.4im) .* Matrix(H)) * psi0

    imbalance = Hamiltonian(
        basis,
        [
            OperatorTerm(
                "z",
                [(2(-1)^(site + 1) / L, site) for site in 1:L],
            ),
        ];
        static_fmt=:csc,
    )
    @test psi_lanczos ≈ psi_exact atol=2e-11
    @test norm(psi_lanczos) ≈ 1.0 atol=2e-13
    @test real(expt_value(imbalance, psi0)) ≈ 1.0 atol=2e-15
    @test abs(real(expt_value(imbalance, psi_lanczos))) < 1.0
end

@testset "paper workflow: sparse periodically driven spin chain" begin
    # Weinberg and Bukov, SciPost Phys. 2, 003 (2017), arXiv:1610.03042,
    # example (iii): heating in a periodically driven transverse-field Ising
    # chain. Both stored pieces and both sampled Hamiltonians remain CSC.
    L = 6
    J, g, h, Omega = 1.0, 0.809, 0.9045, 4.5
    period = 2π / Omega
    basis = SpinBasis1D(L)
    bonds = [(J, site, mod1(site + 1, L)) for site in 1:L]
    z_field = [(h, site) for site in 1:L]
    x_field = [(g, site) for site in 1:L]
    drive(time, frequency) = sign(cos(frequency * time))
    static = Any[
        Any["zz", bonds],
        Any["z", z_field],
        Any["x", x_field],
    ]
    dynamic = Any[
        Any["zz", bonds, drive, (Omega,)],
        Any["z", z_field, drive, (Omega,)],
        Any["x", [(-value, site) for (value, site) in x_field], drive, (Omega,)],
    ]
    H = Hamiltonian(
        static,
        dynamic;
        basis,
        dtype=Float64,
        static_fmt=:csc,
        dynamic_fmt=:csc,
    )

    @test H.data isa SparseMatrixCSC
    @test all(first(term) isa SparseMatrixCSC for term in H.dynamic_terms)
    sample_times = [eps(Float64), period / 2 + eps(Float64)]
    sampled = [tocsc(H; time) for time in sample_times]
    @test all(matrix isa SparseMatrixCSC for matrix in sampled)

    floquet = Floquet(
        Dict(
            :H => H,
            :t_list => sample_times,
            :dt_list => [period / 2, period / 2],
        );
        UF=true,
        VF=true,
        force_ONB=true,
    )
    expected_UF = exp((-im * period / 2) .* Matrix(sampled[2])) *
        exp((-im * period / 2) .* Matrix(sampled[1]))
    identity_matrix = Matrix{ComplexF64}(I, length(basis), length(basis))
    @test floquet.UF ≈ expected_UF atol=3e-13
    @test floquet.UF' * floquet.UF ≈ identity_matrix atol=5e-13
    @test floquet.VF' * floquet.VF ≈ identity_matrix atol=2e-12
    @test all(abs.(floquet.EF) .<= π / period + 2e-12)
end
