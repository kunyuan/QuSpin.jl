@testset "CSR and DIA storage contracts" begin
    basis = SpinBasis1D(7; nup=3, pauli=false)
    terms = [
        OperatorTerm("zz", [
            (0.25 + 0.1site, site, site + 1)
            for site in 1:6
        ]),
        OperatorTerm("z", [((-1)^site * 0.07, site) for site in 1:7]),
    ]
    dense = Hamiltonian(basis, terms; static_fmt=:dense)
    csc = Hamiltonian(basis, terms; static_fmt=:csc)
    csr = Hamiltonian(basis, terms; static_fmt=:csr)
    dia = Hamiltonian(basis, terms; static_fmt=:dia)

    @test csr.data isa SparseMatrixCSR
    @test dia.data isa DIAMatrix
    @test length(csr.data.rowptr) == length(basis) + 1
    @test first(csr.data.rowptr) == 1
    @test last(csr.data.rowptr) == nnz(csr.data) + 1
    @test all(
        issorted(csr.data.colval[
            csr.data.rowptr[row]:(csr.data.rowptr[row + 1] - 1)
        ])
        for row in 1:length(basis)
    )
    @test Matrix(csr) == Matrix(csc) == Matrix(dia) == Matrix(dense)
    @test sparse(csr.data) == csc.data
    @test sparse(dia.data) == csc.data
    @test nnz(csr.data) == nnz(csc.data) == nnz(dia.data)

    vector = ComplexF64[
        sin(0.23index) + im * cos(0.17index)
        for index in 1:length(basis)
    ]
    columns = hcat(vector, reverse(vector))
    @test csr.data * vector ≈ csc.data * vector atol=2e-14
    @test dia.data * vector ≈ csc.data * vector atol=2e-14
    @test csr.data * columns ≈ csc.data * columns atol=2e-14
    @test dia.data * columns ≈ csc.data * columns atol=2e-14

    dimension = length(basis)
    diagonal = collect(range(0.5, 1.5; length=dimension))
    off_diagonal = collect(range(-0.2, 0.2; length=dimension - 1))
    structured_right = (
        Bidiagonal(diagonal, off_diagonal, :U),
        Diagonal(diagonal),
        SymTridiagonal(diagonal, off_diagonal),
        Tridiagonal(off_diagonal, diagonal, reverse(off_diagonal)),
        UpperTriangular(reshape(
            range(-0.2, 0.4; length=dimension^2),
            dimension,
            dimension,
        )),
    )
    for right in structured_right
        expected_product = csc.data * Matrix(right)
        @test csr.data * right ≈ expected_product atol=2e-14
        @test dia.data * right ≈ expected_product atol=2e-14

        csr_output = similar(expected_product)
        dia_output = similar(expected_product)
        mul!(csr_output, csr.data, right)
        mul!(dia_output, dia.data, right)
        @test csr_output ≈ expected_product atol=2e-14
        @test dia_output ≈ expected_product atol=2e-14

        fill!(csr_output, 0.25)
        fill!(dia_output, 0.25)
        mul!(csr_output, csr.data, right, 0.7, -0.3)
        mul!(dia_output, dia.data, right, 0.7, -0.3)
        expected_scaled = 0.7expected_product .- 0.075
        @test csr_output ≈ expected_scaled atol=2e-14
        @test dia_output ≈ expected_scaled atol=2e-14
    end
    @test Matrix(adjoint(csr.data)) == Matrix(adjoint(csc.data))
    @test Matrix(transpose(dia.data)) == Matrix(transpose(csc.data))

    @test update_matrix_formats!(dense, :csr) === dense
    @test dense.data isa SparseMatrixCSR
    @test update_matrix_formats!(dense, :dia) === dense
    @test dense.data isa DIAMatrix
    @test update_matrix_formats!(dense, :csc) === dense
    @test dense.data isa SparseMatrixCSC

    drive(time, frequency) = cos(frequency * time)
    dynamic = Hamiltonian(
        [["z", [(0.2, site) for site in 1:7]]],
        [["zz", [(0.3, 2, 3)], drive, (1.4,)]];
        basis,
        dtype=ComplexF64,
        static_fmt=:csr,
        dynamic_fmt=:dia,
    )
    @test dynamic.static isa SparseMatrixCSR
    @test first(dynamic.dynamic_terms)[1] isa DIAMatrix
    expected = Matrix(dynamic.data) +
        drive(0.37, 1.4) * Matrix(first(dynamic.dynamic_terms)[1])
    @test toarray(dynamic; time=0.37) ≈ expected atol=2e-14
end
