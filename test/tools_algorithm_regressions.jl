using Test
using LinearAlgebra
using SparseArrays
using QuSpin

@testset "Tools algorithm regressions" begin
    @testset "sparse projection and exponential workspace" begin
        observable = spdiagm(0 => [1.0, 2.0, 3.0])
        projector = sparse([1, 2], [1, 2], ComplexF64[1, 1], 3, 2)
        projected = project_op(observable, projector)["Proj_Obs"]
        @test issparse(projected)
        @test Matrix(projected) == ComplexF64[1 0; 0 2]

        generator = [0.0 -1.0; 1.0 0.0]
        action = ExpmMultiplyParallel(generator, 0.25)
        workspace = zeros(ComplexF64, 4)
        result = apply(action, [1.0, 0.0]; work_array=workspace)
        @test result ≈ [cos(0.25), sin(0.25)] atol=3e-16
        result[1] = 7
        @test workspace[3] == 7
    end

    @testset "streaming evolution and observations" begin
        calls = Ref(0)
        derivative = function (time, state)
            calls[] += 1
            return -im .* state
        end
        iterator = evolve(
            ComplexF64[1],
            0.0,
            [0.0, 0.2],
            derivative;
            max_step=0.01,
            iterate=true,
        )
        first_state, iterator_state = iterate(iterator)
        @test first_state == ComplexF64[1]
        @test calls[] == 0
        second_state, _ = iterate(iterator, iterator_state)
        @test calls[] > 0
        @test only(second_state) ≈ exp(-0.2im) atol=2e-10

        inplace_calls = Ref(0)
        inplace_derivative = function (destination, time, state, rate)
            inplace_calls[] += 1
            @. destination = rate * state
            return destination
        end
        adaptive = evolve(
            ComplexF64[1],
            0.0,
            [2.0],
            inplace_derivative;
            f_params=(-0.7im,),
            max_step=0.5,
            rtol=1e-10,
            atol=1e-12,
        )
        @test only(adaptive) ≈ exp(-1.4im) atol=2e-10
        @test inplace_calls[] > 0
        fixed = evolve(
            ComplexF64[1],
            0.0,
            [0.2],
            derivative;
            solver_name=:rk4,
            max_step=0.01,
        )
        @test only(fixed) ≈ exp(-0.2im) atol=2e-10
        @test_throws ArgumentError evolve(
            [1.0],
            0.0,
            [0.1],
            (time, state) -> state;
            solver_name=:unknown,
        )

        times = [0.0, 0.25, 0.5]
        states = (
            ComplexF64[cos(time), sin(time)]
            for time in times
        )
        observed = obs_vs_time(
            states,
            times,
            Dict("z" => Diagonal([1.0, -1.0])),
        )
        @test real.(observed["z"]) ≈ cos.(2 .* times) atol=2e-16
    end

    @testset "time-ordered and sparse Floquet evolution" begin
        Z = ComplexF64[1 0; 0 -1]
        X = ComplexF64[0 1; 1 0]
        hamiltonian = time -> cos(time) .* Z .+ sin(time) .* X
        period = 0.4
        floquet = Floquet(
            Dict(
                :H => hamiltonian,
                :T => period,
                :max_step => 0.001,
            );
            HF=true,
            UF=true,
        )

        reference = Matrix{ComplexF64}(I, 2, 2)
        steps = 4000
        step = period / steps
        for index in 1:steps
            midpoint = (index - 0.5) * step
            reference =
                exp((-im * step) .* hamiltonian(midpoint)) * reference
        end
        @test floquet.UF ≈ reference atol=2e-6
        @test exp((-im * period) .* floquet.HF) ≈ floquet.UF atol=2e-6

        dimension = 66
        diagonal = collect(range(-1.0, 1.0; length=dimension))
        sparse_floquet = Floquet(
            Dict(:H => spdiagm(0 => diagonal), :T => 0.2);
            UF=true,
        )
        expected = Diagonal(exp.((-0.2im) .* diagonal))
        @test sparse_floquet.UF ≈ expected atol=1e-10
    end

    @testset "sparse blocks and cache controls" begin
        terms = [OperatorTerm("z", [(1.0, 1), (0.5, 2)])]
        blocks = [Dict(:nup => sector) for sector in 0:2]
        projector, block_hamiltonian = block_diag_hamiltonian(
            blocks,
            terms,
            Any[],
            SpinBasis1D,
            (2,),
            ComplexF64;
            basis_kwargs=Dict(:pauli => false),
        )
        @test issparse(projector)
        @test issparse(block_hamiltonian)
        dense_projector, _ = block_diag_hamiltonian(
            blocks,
            terms,
            Any[],
            SpinBasis1D,
            (2,),
            ComplexF64;
            basis_kwargs=Dict(:pauli => false),
            get_proj_kwargs=Dict(:sparse => false),
        )
        @test !issparse(dense_projector)
        @test_throws ArgumentError BlockOps(
            blocks,
            terms,
            Any[],
            SpinBasis1D,
            (2,),
            ComplexF64;
            get_proj_kwargs=Dict(:unsupported => true),
        )

        operator = BlockOps(
            blocks,
            terms,
            Any[],
            SpinBasis1D,
            (2,),
            ComplexF64;
            basis_kwargs=Dict(:pauli => false),
            save_previous_data=false,
        )
        state = normalize(ComplexF64[1, 2, 3, 4])
        iterator = evolve(
            operator,
            state,
            0.0,
            [0.0, 0.2];
            iterate=true,
            block_diag=true,
            n_jobs=2,
        )
        states = collect(iterator)
        @test states[1] ≈ state atol=3e-16
        @test isempty(operator.H_dict)
        @test isempty(operator.P_dict)

        unchecked = BlockOps(
            [Dict()],
            [OperatorTerm("+", [(1.0, 1)])],
            Any[],
            SpinBasis1D,
            (2,),
            ComplexF64;
            check_herm=false,
            check_pcon=false,
            compute_all_blocks=true,
        )
        @test length(unchecked.H_dict) == 1
    end

    @testset "direct reduced ensemble and streaming Lanczos" begin
        basis = SpinBasis1D(2)
        state = ComplexF64[inv(sqrt(2)), 0, 0, inv(sqrt(2))]
        ensemble = diag_ensemble(
            2,
            state,
            [-2.0, -1.0, 1.0, 2.0],
            Matrix{ComplexF64}(I, 4, 4);
            density=false,
            Srdm_Renyi=true,
            Srdm_args=Dict(:basis => basis, :sub_sys_A => [1]),
        )
        @test ensemble["Srdm_pure"] ≈ log(2) atol=3e-16

        matrix = [
            1.0 0.2 0.0 0.0
            0.2 2.0 0.3 0.0
            0.0 0.3 3.0 0.4
            0.0 0.0 0.4 4.0
        ]
        initial = [1.0, 2.0, -1.0, 0.5]
        energies, vectors, rows = lanczos_iter(matrix, initial, 3)
        collected_once = collect(rows)
        @test collect(rows) ≈ collected_once atol=1e-14
        output = zeros(4)
        @test lin_comb_Q_T([1.0, -2.0, 0.5], rows; out=output) === output

        full_rows = reduce(vcat, permutedims.(collected_once))
        observables = Dict("identity" => Matrix{Float64}(I, 4, 4))
        ftlm_stream, ftlm_identity_stream =
            ftlm_static_iteration(
                observables,
                energies,
                vectors,
                rows;
                beta=[0.5, 2.0],
            )
        ftlm_dense, ftlm_identity_dense =
            ftlm_static_iteration(
                observables,
                energies,
                vectors,
                full_rows;
                beta=[0.5, 2.0],
            )
        @test ftlm_stream["identity"] ≈ ftlm_dense["identity"] atol=2e-15
        @test ftlm_identity_stream ≈ ftlm_identity_dense atol=2e-15

        ltlm_stream, ltlm_identity_stream =
            ltlm_static_iteration(
                observables,
                energies,
                vectors,
                rows;
                beta=[2.0, 5.0],
            )
        ltlm_dense, ltlm_identity_dense =
            ltlm_static_iteration(
                observables,
                energies,
                vectors,
                full_rows;
                beta=[2.0, 5.0],
            )
        @test ltlm_stream["identity"] ≈ ltlm_dense["identity"] atol=2e-15
        @test ltlm_identity_stream ≈ ltlm_identity_dense atol=2e-15
    end
end
