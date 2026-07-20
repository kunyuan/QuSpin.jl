@testset "general-basis Hamiltonian semantics" begin
    bosons = BosonBasis1D(2; Nb=1, sps=3)
    boson_hopping = [
        OperatorTerm("+-", [(-1.0, 1, 2)]),
        OperatorTerm("-+", [(-1.0, 1, 2)]),
    ]
    H_boson = Hamiltonian(bosons, boson_hopping; static_fmt=:csc)
    @test H_boson.basis === bosons
    @test H_boson.static isa SparseMatrixCSC
    @test Matrix(H_boson) ≈ [0 -1; -1 0]
    @test sort(eigvals(H_boson)) ≈ [-1, 1]

    fermions = SpinlessFermionBasis1D(3; Nf=1)
    fermion_hopping = [
        OperatorTerm("+-", [(-1.0, 1, 2), (-1.0, 2, 3)]),
        OperatorTerm("-+", [(1.0, 1, 2), (1.0, 2, 3)]),
    ]
    H_fermion = Hamiltonian(fermions, fermion_hopping; static_fmt=:csc)
    @test Matrix(H_fermion) ≈ [
        0 -1 0
        -1 0 -1
        0 -1 0
    ]

    spinful = SpinfulFermionBasis1D(2; Nf=(1, 1))
    hubbard_terms = [
        OperatorTerm("+-|", [(-1.0, 1, 2)]),
        OperatorTerm("-+|", [(1.0, 1, 2)]),
        OperatorTerm("|+-", [(-1.0, 1, 2)]),
        OperatorTerm("|-+", [(1.0, 1, 2)]),
        OperatorTerm("n|n", [(2.0, 1, 1), (2.0, 2, 2)]),
    ]
    H_hubbard = Hamiltonian(spinful, hubbard_terms; static_fmt=:csc)
    @test sort(real.(eigvals(H_hubbard))) ≈
        sort([0.0, 2.0, 1 - sqrt(5), 1 + sqrt(5)]) atol=1e-12
    @test norm(H_hubbard * ones(ComplexF64, length(spinful))) > 0

    X = ComplexF64[0 1; 1 0]
    custom = UserBasis(UInt64, 2, Dict('x' => X))
    H_custom = Hamiltonian(
        custom,
        [OperatorTerm("x", [(1.0, 1), (1.0, 2)])],
    )
    @test sort(real.(eigvals(H_custom))) ≈ [-2, 0, 0, 2]
end

@testset "honest dense CSC CSR and DIA storage" begin
    basis = SpinBasis1D(4; nup=2, pauli=false)
    terms = [OperatorTerm("zz", [(1.0, 1, 2), (1.0, 2, 3)])]
    dense = Hamiltonian(basis, terms; static_fmt=:dense)
    csc = Hamiltonian(basis, terms; static_fmt=:csc)
    csr = Hamiltonian(basis, terms; static_fmt=:csr)
    dia = Hamiltonian(basis, terms; static_fmt=:dia)

    @test dense.static isa Matrix
    @test csc.static isa SparseMatrixCSC
    @test csr.static isa SparseMatrixCSR
    @test dia.static isa DIAMatrix
    @test Matrix(csc.static) == Matrix(csr.static) == Matrix(dia.static) ==
        Matrix(dense.static)
    @test csr * ones(length(basis)) == dense * ones(length(basis))
    @test dia * ones(length(basis)) == dense * ones(length(basis))
    @test tocsr(csc) isa SparseMatrixCSR
    @test Matrix(tocsr(csc)) == Matrix(dense)
    @test as_sparse_format(dense; static_fmt=:csr).static isa SparseMatrixCSR
    @test as_sparse_format(dense; static_fmt=:dia).static isa DIAMatrix

    operator = QuantumOperator(
        basis,
        Dict(:z => terms);
        matrix_formats=Dict(:z => :csr),
    )
    @test get_operators(operator, :z) isa SparseMatrixCSR
    @test tocsr(operator; pars=Dict(:z => 1.0)) isa SparseMatrixCSR
end

@testset "physical symmetry sectors and validation" begin
    L = 4
    periodic_sites = [(i, mod1(i + 1, L)) for i in 1:L]
    spin_terms = [
        OperatorTerm("zz", [(1.0, sites...) for sites in periodic_sites]),
        OperatorTerm("+-", [(0.5, sites...) for sites in periodic_sites]),
        OperatorTerm("-+", [(0.5, sites...) for sites in periodic_sites]),
    ]
    full_spin = SpinBasis1D(L; nup=2, pauli=false)
    full_spectrum = sort(real.(eigvals(Hamiltonian(full_spin, spin_terms))))
    momentum_spectrum = sort(vcat([
        real.(eigvals(Hamiltonian(
            SpinBasis1D(L; nup=2, pauli=false, kblock=momentum),
            spin_terms,
        )))
        for momentum in 0:(L - 1)
    ]...))
    @test momentum_spectrum ≈ full_spectrum atol=2e-12

    parity_spectrum = sort(vcat([
        real.(eigvals(Hamiltonian(
            SpinBasis1D(L; nup=2, pauli=false, pblock=parity),
            spin_terms,
        )))
        for parity in (-1, 1)
    ]...))
    inversion_spectrum = sort(vcat([
        real.(eigvals(Hamiltonian(
            SpinBasis1D(L; nup=2, pauli=false, zblock=parity),
            spin_terms,
        )))
        for parity in (-1, 1)
    ]...))
    @test parity_spectrum ≈ full_spectrum atol=2e-12
    @test inversion_spectrum ≈ full_spectrum atol=2e-12

    momentum_basis = SpinBasis1D(L; nup=2, pauli=false, kblock=1)
    projector = projection_matrix(momentum_basis, ComplexF64)
    @test projector' * projector ≈ I atol=3e-13
    @test size(projector) == (2^L, length(momentum_basis))
    represented = first(momentum_basis.symmetry.parent_states)
    if any(!iszero, momentum_basis.symmetry.projector[
        momentum_basis.symmetry.parent_lookup[represented],
        :,
    ])
        @test representative(momentum_basis, represented) in momentum_basis.states
        @test normalization(momentum_basis, represented) >= 1
    end

    bad_translation = [["z", [(1.0, 1)]]]
    @test !check_symm(momentum_basis, bad_translation)
    @test_throws ArgumentError Hamiltonian(
        bad_translation,
        Any[];
        basis=momentum_basis,
        dtype=ComplexF64,
    )
    @test !check_pcon(full_spin, [["+", [(1.0, 1)]]])
    @test_throws ArgumentError Hamiltonian(
        [["+", [(1.0, 1)]]],
        Any[];
        basis=full_spin,
        dtype=ComplexF64,
        check_herm=false,
    )
    @test !check_hermitian(SpinBasis1D(2), [["+", [(1.0, 1)]]])

    hopping_plus = [(-1.0, sites...) for sites in periodic_sites]
    hopping_minus = [(1.0, sites...) for sites in periodic_sites]
    fermion_terms = [
        OperatorTerm("+-", hopping_plus),
        OperatorTerm("-+", hopping_minus),
    ]
    full_fermion = SpinlessFermionBasis1D(L; Nf=2)
    fermion_spectrum = sort(real.(eigvals(Hamiltonian(
        full_fermion,
        fermion_terms,
    ))))
    fermion_blocks = sort(vcat([
        real.(eigvals(Hamiltonian(
            SpinlessFermionBasis1D(L; Nf=2, kblock=momentum),
            fermion_terms,
        )))
        for momentum in 0:(L - 1)
    ]...))
    @test fermion_blocks ≈ fermion_spectrum atol=2e-12

    boson_terms = [
        OperatorTerm("+-", hopping_plus),
        OperatorTerm("-+", hopping_plus),
    ]
    full_boson = BosonBasis1D(L; Nb=2, sps=3)
    boson_spectrum = sort(real.(eigvals(Hamiltonian(full_boson, boson_terms))))
    boson_blocks = sort(vcat([
        real.(eigvals(Hamiltonian(
            BosonBasis1D(L; Nb=2, sps=3, kblock=momentum),
            boson_terms,
        )))
        for momentum in 0:(L - 1)
    ]...))
    @test boson_blocks ≈ boson_spectrum atol=3e-12

    spinful_terms = [
        OperatorTerm("+-|", hopping_plus),
        OperatorTerm("-+|", hopping_minus),
        OperatorTerm("|+-", hopping_plus),
        OperatorTerm("|-+", hopping_minus),
    ]
    full_spinful = SpinfulFermionBasis1D(L; Nf=(1, 1))
    spinful_spectrum = sort(real.(eigvals(Hamiltonian(
        full_spinful,
        spinful_terms,
    ))))
    spinful_blocks = sort(vcat([
        real.(eigvals(Hamiltonian(
            SpinfulFermionBasis1D(L; Nf=(1, 1), kblock=momentum),
            spinful_terms,
        )))
        for momentum in 0:(L - 1)
    ]...))
    spin_exchange_blocks = sort(vcat([
        real.(eigvals(Hamiltonian(
            SpinfulFermionBasis1D(L; Nf=(1, 1), sblock=parity),
            spinful_terms,
        )))
        for parity in (-1, 1)
    ]...))
    @test spinful_blocks ≈ spinful_spectrum atol=4e-12
    @test spin_exchange_blocks ≈ spinful_spectrum atol=4e-12
end

@testset "matrix-free Krylov paths" begin
    L = 8
    basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
    bonds = [(i, mod1(i + 1, L)) for i in 1:L]
    terms = [
        OperatorTerm("zz", [(0.9, sites...) for sites in bonds]),
        OperatorTerm("+-", [(0.45, sites...) for sites in bonds]),
        OperatorTerm("-+", [(0.45, sites...) for sites in bonds]),
    ]
    linear = QuantumLinearOperator(basis, terms)
    sparse_H = Hamiltonian(basis, terms; static_fmt=:csc)
    vector = normalize(ComplexF64.(
        sin.(1:length(basis)) .+ im .* cos.(2 .* (1:length(basis))),
    ))
    @test linear.explicit_data === nothing
    @test linear * vector ≈ sparse_H * vector atol=3e-14
    @test Matrix(linear) ≈ Matrix(sparse_H) atol=2e-14
    values, vectors = eigsh(linear; k=2, which=:SA)
    @test linear * vectors ≈ vectors * Diagonal(values) atol=3e-11
    @test linear.explicit_data === nothing

    fermions = SpinlessFermionBasis1D(5; Nf=2)
    open_bonds = [(i, i + 1) for i in 1:4]
    fermion_terms = [
        OperatorTerm("+-", [(-0.7, sites...) for sites in open_bonds]),
        OperatorTerm("-+", [(0.7, sites...) for sites in open_bonds]),
    ]
    fermion_linear = QuantumLinearOperator(fermions, fermion_terms)
    fermion_matrix = Hamiltonian(fermions, fermion_terms; static_fmt=:csc)
    fermion_vector = normalize(ComplexF64.(1:length(fermions)))
    @test fermion_linear * fermion_vector ≈
        fermion_matrix * fermion_vector atol=2e-14

    times = [0.0, 0.17, 0.83]
    evolved = evolve(sparse_H, vector, 0.0, times)
    @test evolved[:, end] ≈
        exp(-im * times[end] * Matrix(sparse_H)) * vector atol=3e-11
    @test all(
        isapprox(norm(column), 1.0; atol=3e-12)
        for column in eachcol(evolved)
    )

    exponential = ExpmMultiplyParallel(sparse_H.data, -0.37im)
    @test exponential.A === sparse_H.data
    @test exponential * vector ≈
        exp(-0.37im * Matrix(sparse_H)) * vector atol=3e-11

    floquet = Floquet(
        Dict(:H => sparse_H, :T => 0.41);
        UF=true,
        thetaF=true,
    )
    @test floquet.UF' * floquet.UF ≈ I atol=5e-11
    @test abs.(floquet.thetaF) ≈ ones(length(basis)) atol=5e-11
end
