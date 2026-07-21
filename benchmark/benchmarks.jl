using BenchmarkTools
using LinearAlgebra
using QuSpin
using SparseArrays

BLAS.set_num_threads(1)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20

function xxz_terms(L; delta=0.7)
    bonds = [(1.0, i, mod1(i + 1, L)) for i in 1:L]
    return [
        OperatorTerm("+-", [(0.5, i, j) for (_, i, j) in bonds]),
        OperatorTerm("-+", [(0.5, i, j) for (_, i, j) in bonds]),
        OperatorTerm("zz", [(delta, i, j) for (_, i, j) in bonds]),
    ]
end

function benchmark_suite()
    L = 14
    terms = xxz_terms(L)
    basis = SpinBasis1D(L; nup=L ÷ 2, pauli=false)
    H = Hamiltonian(basis, terms; static_fmt=:csc)
    vector = normalize!(ones(Float64, length(basis)))

    values, vectors = eigsh(H; k=4, which=:SA, v0=vector, tol=1e-10)
    @assert H.data isa SparseMatrixCSC
    @assert norm(H.data * vectors - vectors * Diagonal(values)) < 1e-7

    constrained_length = 18
    constrained_states = constraint_states(
        constrained_length;
        prefix_allowed=(occupations, site) ->
            site == 1 ||
            occupations[site - 1] + occupations[site] <= 1,
        state_allowed=occupations ->
            occupations[1] + occupations[end] <= 1,
    )
    constrained_user = UserBasis(
        UInt64,
        constrained_length,
        Dict('x' => ComplexF64[0 1; 1 0]);
        states=constrained_states,
        allowed_ops=('x',),
    )
    constrained_terms = [
        OperatorTerm(
            "x",
            [(1.0, site) for site in 1:constrained_length],
        ),
    ]
    constrained_matrix = Hamiltonian(
        constrained_user,
        constrained_terms;
        static_fmt=:csc,
        check_symm=false,
        check_herm=false,
        check_pcon=false,
    )
    @assert length(constrained_user) == 5_778
    @assert constrained_matrix.data isa SparseMatrixCSC
    @assert ishermitian(constrained_matrix.data)

    nonbinary_length = 7
    nonbinary_user = UserBasis(
        UInt64,
        nonbinary_length,
        Dict('q' => ComplexF64[0 1 0; 1 0 1; 0 1 0]);
        sps=3,
        states=UInt64.(0:(3^nonbinary_length - 1)),
        allowed_ops=('q',),
    )
    nonbinary_terms = [
        OperatorTerm(
            "q",
            [(1.0, site) for site in 1:nonbinary_length],
        ),
    ]
    nonbinary_matrix = Hamiltonian(
        nonbinary_user,
        nonbinary_terms;
        static_fmt=:csc,
        check_symm=false,
        check_herm=false,
        check_pcon=false,
    )
    @assert length(nonbinary_user) == 3^nonbinary_length
    @assert nonbinary_matrix.data isa SparseMatrixCSC
    @assert ishermitian(nonbinary_matrix.data)

    return Dict(
        "fixed-weight basis construction" => @benchmarkable(
            SpinBasis1D($L; nup=$L ÷ 2, pauli=false)
        ),
        "CSC Hamiltonian construction" => @benchmarkable(
            Hamiltonian($basis, $terms; static_fmt=:csc)
        ),
        "constrained UserBasis CSC construction" => @benchmarkable(
            Hamiltonian(
                $constrained_user,
                $constrained_terms;
                static_fmt=:csc,
                check_symm=false,
                check_herm=false,
                check_pcon=false,
            )
        ),
        "nonbinary UserBasis CSC construction" => @benchmarkable(
            Hamiltonian(
                $nonbinary_user,
                $nonbinary_terms;
                static_fmt=:csc,
                check_symm=false,
                check_herm=false,
                check_pcon=false,
            )
        ),
        "CSC matrix-vector action" => @benchmarkable($H.data * $vector),
        "ARPACK lowest eigenpairs" => @benchmarkable(
            eigsh($H; k=4, which=:SA, v0=$vector, tol=1e-10)
        ),
    )
end

println("QuSpin.jl benchmark suite")
println("Julia threads: ", Threads.nthreads(), "; BLAS threads: ", BLAS.get_num_threads())
for (name, benchmark) in sort!(collect(benchmark_suite()); by=first)
    trial = run(benchmark)
    estimate = median(trial)
    println()
    println(name)
    show(stdout, MIME("text/plain"), estimate)
    println()
end
