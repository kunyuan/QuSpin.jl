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

    return Dict(
        "fixed-weight basis construction" => @benchmarkable(
            SpinBasis1D($L; nup=$L ÷ 2, pauli=false)
        ),
        "CSC Hamiltonian construction" => @benchmarkable(
            Hamiltonian($basis, $terms; static_fmt=:csc)
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
