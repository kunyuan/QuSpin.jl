using Test
using LinearAlgebra
using Random
using SparseArrays
using QuSpin

function _qlo_buffer_allocations(operator, vector, matrix)
    vector_output = similar(vector)
    matrix_output = similar(matrix)
    apply(operator, vector; out=vector_output)
    mul!(matrix_output, operator, matrix)
    return (
        @allocated(apply(operator, vector; out=vector_output)),
        @allocated(mul!(matrix_output, operator, matrix)),
    )
end

@testset "operator algorithm regressions" begin
    @testset "linear sparse conversions and transforms" begin
        Random.seed!(0x51a7)
        source = sprand(ComplexF64, 11, 7, 0.28)
        csr = SparseMatrixCSR(source)
        @test sparse(csr) == source
        @test Matrix(csr) == Matrix(source)
        @test Matrix(transpose(csr)) == transpose(Matrix(source))
        @test Matrix(adjoint(csr)) == adjoint(Matrix(source))
        @test Matrix(conj(csr)) == conj(Matrix(source))

        dia_source = spdiagm(
            7,
            9,
            -2 => ComplexF64[1, 2, 3, 4, 5],
            1 => ComplexF64[1im, 2im, 3im, 4im, 5im, 6im, 7im],
        )
        dia = DIAMatrix(dia_source)
        @test sparse(dia) == dia_source
        @test Matrix(transpose(dia)) == transpose(Matrix(dia_source))
        @test Matrix(adjoint(dia)) == adjoint(Matrix(dia_source))
    end

    @testset "matrix-free QLO transforms, diagonal, and matrix RHS" begin
        basis = SpinBasis1D(3; pauli=false)
        terms = [
            OperatorTerm("z", [(0.2, 1), (0.3, 1)]),
            OperatorTerm("+-", [(0.4, 1, 2)]),
            OperatorTerm("-+", [(0.4, 1, 2)]),
        ]
        qlo = QuantumLinearOperator(basis, terms)
        reference = sum(
            operator_matrix(basis, term.op, term.couplings)
            for term in terms
        )
        @test qlo.static_list == terms
        @test isempty(filter(term -> term.op == "z", qlo.action_terms))
        @test qlo.extracted_diagonal == diag(
            operator_matrix(basis, "z", [(0.5, 1)]),
        )

        vector = randn(ComplexF64, length(basis))
        rhs = randn(ComplexF64, length(basis), 4)
        output = similar(rhs)
        @test apply(qlo, vector) ≈ reference * vector
        @test apply(qlo, vector; out=similar(vector)) ≈ reference * vector
        @test mul!(output, qlo, rhs) ≈ reference * rhs
        @test eltype(output) === ComplexF64
        vector_bytes, matrix_bytes =
            _qlo_buffer_allocations(qlo, vector, rhs)
        @test vector_bytes <= 512
        @test matrix_bytes <= 4_096
        @test transpose(qlo) * vector ≈ transpose(reference) * vector
        @test conj(qlo) * vector ≈ conj(reference) * vector
        @test adjoint(qlo) * vector ≈ adjoint(reference) * vector

        rows = randn(ComplexF64, 5, length(basis))
        @test right_apply(qlo, rows) ≈ rows * reference

        reduced_basis = SpinBasis1D(
            4;
            nup=2,
            pauli=false,
            kblock=0,
        )
        nonsymmetric_terms = [OperatorTerm("z", [(0.7, 1)])]
        unchecked_qlo = QuantumLinearOperator(
            reduced_basis,
            nonsymmetric_terms;
            check_symm=false,
        )
        unchecked_reference = operator_matrix(
            reduced_basis,
            "z",
            [(0.7, 1)],
        )
        reduced_vector = randn(ComplexF64, length(reduced_basis))
        @test !isempty(unchecked_qlo.action_terms)
        @test unchecked_qlo * reduced_vector ≈
            unchecked_reference * reduced_vector atol=2e-14
    end

    @testset "Krylov ExpOp action, shift, grid, and laziness" begin
        basis = SpinBasis1D(3; pauli=false)
        H = Hamiltonian(
            basis,
            [
                OperatorTerm("x", [(0.7, 1), (-0.2, 2)]),
                OperatorTerm("z", [(0.4, 3)]),
            ],
        )
        matrix = Matrix(H)
        vector = randn(ComplexF64, length(basis))
        exponential = ExpOp(
            H;
            a=-0.3im,
            start=0.0,
            stop=1.0,
            num=4,
            iterate=true,
        )
        lazy_states = apply(exponential, vector; shift=0.15)
        @test !(lazy_states isa AbstractArray)
        states = collect(lazy_states)
        for (state, time) in zip(states, exponential.grid)
            @test state ≈
                exp(-0.3im * time * (matrix + 0.15I)) * vector atol=2e-11
        end
        set_iterate!(exponential, false)
        table = apply(exponential, vector; shift=0.15)
        @test table ≈ reduce(hcat, states) atol=2e-11
    end

    @testset "component actions for dynamic density evolution" begin
        basis = SpinBasis1D(1; pauli=false)
        Z = ComplexF64[-0.5 0; 0 0.5]
        X = ComplexF64[0 0.5; 0.5 0]
        H = Hamiltonian(
            Any[Z],
            Any[Any[X, (time,) -> 1.0, ()]];
            basis,
            dtype=ComplexF64,
        )
        rho = ComplexF64[1 0; 0 0]
        time = 0.3
        evolved = evolve(
            H,
            rho,
            0.0,
            [time];
            eom=:LvNE,
            max_step=0.2,
            rtol=1e-11,
            atol=1e-13,
        )[:, :, 1]
        unitary = exp(-im * time * (Z + X))
        @test evolved ≈ unitary * rho * unitary' atol=2e-10

        qo = QuantumOperator(basis, Dict(:z => Z, :x => X))
        pars = Dict(:z => 0.7, :x => -0.4)
        rows = randn(ComplexF64, 3, 2)
        @test right_apply(qo, rows; pars) ≈
            rows * (0.7Z - 0.4X)
        @test aslinearoperator(qo; pars) * vec(rho[:, 1]) ≈
            (0.7Z - 0.4X) * vec(rho[:, 1])
    end
end
