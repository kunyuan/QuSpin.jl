# Benchmarks

QuSpin.jl keeps correctness and performance separate: physical invariants and
oracle comparisons are CI gates, while hosted-runner timings are observational
because shared hardware is noisy.

## Reproduce locally

The public benchmark environment uses `BenchmarkTools` and fixes Julia to one
thread unless the caller explicitly changes it:

```sh
julia --project=benchmark -e '
    using Pkg
    Pkg.develop(path=pwd())
    Pkg.instantiate()
'
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    julia --project=benchmark benchmark/benchmarks.jl
```

Each benchmark performs an untimed correctness preflight. Report medians and
allocations; never compare first-call Julia timings against warmed Python calls.

## Current same-runner snapshot

The following warm medians came from verification run
[29800995942](https://github.com/kunyuan/quspin-julia-verify/actions/runs/29800995942)
on one Julia and BLAS thread. Speedup is Python median divided by Julia median.

| Workload | Python (ms) | Julia (ms) | Julia speedup |
| --- | ---: | ---: | ---: |
| MBL mid-spectrum shift-invert | 394.625 | 132.156 | 2.99x |
| XXZ Lanczos quench | 456.809 | 397.103 | 1.15x |
| Floquet full unitary | 582.782 | 420.390 | 1.39x |
| spinful Hubbard spectrum | 47.736 | 35.684 | 1.34x |
| interacting SSH spectrum | 104.876 | 98.430 | 1.07x |
| translation-sector XXZ | 49.365 | 78.016 | 0.63x |

Julia won five of six workflows in this run; the geometric-mean speedup was
1.275x. This is not a hardware-independent promise. Translation-sector basis
and workflow construction remains a tracked performance target.

## Interpreting results

- `Hamiltonian` construction and solver time should be measured separately.
- Keep storage controlled: CSC-to-CSC and dense-to-dense are meaningful pairs.
- Validate dimensions, fingerprints, residuals, norm, trace, or unitarity
  outside the timed region.
- Record raw samples, package versions, CPU, thread counts, and allocations.
- Treat a speed difference smaller than run-to-run noise as inconclusive.

The generic spinless transition-to-CSC implementation reduced an isolated
local construction kernel from 24.548 ms to 5.968 ms (4.11x). End-to-end SSH
improved less because ARPACK became the dominant stage; this distinction is why
the benchmark suite reports both kernels and complete workflows.
