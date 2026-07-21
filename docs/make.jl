using Documenter
using QuSpin

DocMeta.setdocmeta!(
    QuSpin,
    :DocTestSetup,
    :(using QuSpin, LinearAlgebra, SparseArrays);
    recursive=true,
)

makedocs(
    sitename="QuSpin.jl",
    modules=[QuSpin, QuSpin.Basis, QuSpin.Operators, QuSpin.Tools],
    authors="Matrix Lab contributors",
    format=Documenter.HTML(
        canonical="https://kunyuan.github.io/QuSpin.jl",
        edit_link="main",
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Tutorials" => [
            "XXZ exact diagonalization" => "tutorials/xxz-ed.md",
            "Lanczos quantum quench" => "tutorials/quantum-quench.md",
        ],
        "Cookbook" => "cookbook.md",
        "Benchmarks" => "benchmarks.md",
        "API reference" => "api.md",
    ],
    doctest=true,
    checkdocs=:exports,
    warnonly=false,
)

deploydocs(
    repo="github.com/kunyuan/QuSpin.jl.git",
    devbranch="main",
    push_preview=false,
)
