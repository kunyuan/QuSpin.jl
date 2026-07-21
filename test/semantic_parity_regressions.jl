using Test
using LinearAlgebra
using SparseArrays
using QuSpin

@testset "semantic parity regressions" begin
    @testset "dynamic Hamiltonian algebra keeps every time-dependent term" begin
        basis = SpinBasis1D(2; pauli=false)
        A0 = ComplexF64[0.2 0.3im 0 0; -0.3im -0.1 0.4 0; 0 0.4 0.7 0.1im; 0 0 -0.1im -0.8]
        A1 = ComplexF64[1 0 0 0; 0 -1 0 0; 0 0 0.5 0; 0 0 0 -0.5]
        B0 = ComplexF64[0 0.2 0 0; 0.2 0 0.1im 0; 0 -0.1im 0 0.3; 0 0 0.3 0]
        B1 = ComplexF64[0 0 0.4 0; 0 0 0 -0.2im; 0.4 0 0 0; 0 0.2im 0 0]
        cosine(time, frequency) = cos(frequency * time)
        ramp(time, slope) = slope * time
        H1 = Hamiltonian(
            Any[A0],
            Any[Any[A1, cosine, (1.7,)]];
            basis,
            dtype=ComplexF64,
            check_herm=false,
            check_symm=false,
            check_pcon=false,
        )
        H2 = Hamiltonian(
            Any[B0],
            Any[Any[B1, ramp, (-0.4,)]];
            basis,
            dtype=ComplexF64,
            check_herm=false,
            check_symm=false,
            check_pcon=false,
        )
        U = exp(0.23im .* ComplexF64[0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0])

        for time in (0.0, 0.31, 1.2)
            M1 = toarray(H1; time)
            M2 = toarray(H2; time)
            @test toarray(rotate_by(H1, U); time) ≈ U' * M1 * U atol=2e-14
            @test toarray(project_to(H1, Matrix{ComplexF64}(I, 4, 4)); time) ≈ M1
            @test toarray(H1 + H2; time) ≈ M1 + M2 atol=2e-14
            @test toarray(H1 - H2; time) ≈ M1 - M2 atol=2e-14
            @test toarray(2.5H1; time) ≈ 2.5M1 atol=2e-14
            @test toarray(H1 * H2; time) ≈ M1 * M2 atol=3e-14
            @test toarray(commutator(H1, H2); time) ≈ M1 * M2 - M2 * M1 atol=4e-14
            @test toarray(anti_commutator(H1, H2); time) ≈ M1 * M2 + M2 * M1 atol=4e-14
        end
    end

    @testset "Hamiltonian evolution distinguishes SE batches and LvNE states" begin
        basis = SpinBasis1D(1; pauli=false)
        X = ComplexF64[0 1; 1 0]
        Z = ComplexF64[1 0; 0 -1]
        H = Hamiltonian(
            Any[0.3Z],
            Any[Any[0.7X, (time, omega) -> cos(omega * time), (1.2,)]];
            basis,
            dtype=ComplexF64,
            check_herm=false,
            check_symm=false,
            check_pcon=false,
        )
        times = [0.0, 0.2, 0.7]
        batch = ComplexF64[1 0 1; 0 1 im]
        batch ./= sqrt.(sum(abs2, batch; dims=1))
        evolved_batch = evolve(
            H,
            batch,
            0.0,
            times;
            eom=:SE,
            max_step=0.02,
            rtol=1e-11,
            atol=1e-13,
        )
        @test size(evolved_batch) == (2, 3, 3)
        for column in axes(batch, 2)
            @test evolved_batch[:, column, :] ≈ evolve(
                H,
                batch[:, column],
                0.0,
                times;
                eom=:SE,
                max_step=0.02,
                rtol=1e-11,
                atol=1e-13,
            ) atol=3e-11
        end

        static_H = Hamiltonian(
            Any[0.3Z + 0.7X],
            Any[];
            basis,
            dtype=ComplexF64,
            check_herm=false,
            check_symm=false,
            check_pcon=false,
        )
        static_batch = evolve(static_H, batch, 0.0, times; eom=:SE)
        @test size(static_batch) == (2, 3, 3)
        for column in axes(batch, 2)
            @test static_batch[:, column, :] ≈ evolve(
                static_H,
                batch[:, column],
                0.0,
                times;
                eom=:SE,
            ) atol=3e-12
        end
        @test evolve(static_H, batch, 0.0, 0.2; eom=:SE) ≈
            static_batch[:, :, 2] atol=3e-12
        @test collect(evolve(
            static_H,
            batch,
            0.0,
            times;
            eom=:SE,
            iterate=true,
        )) ≈ [static_batch[:, :, index] for index in axes(static_batch, 3)]
        @test size(evolve(static_H, batch, 0.0, Float64[]; eom=:SE)) ==
            (2, 3, 0)

        rho0 = ComplexF64[1 0; 0 0]
        rho_t = evolve(
            H,
            rho0,
            0.0,
            times;
            eom=:LvNE,
            max_step=0.02,
            rtol=1e-11,
            atol=1e-13,
        )
        @test size(rho_t) == (2, 2, 3)
        @test all(isapprox(tr(rho_t[:, :, index]), 1; atol=2e-12) for index in axes(rho_t, 3))
        @test_throws ArgumentError evolve(H, ComplexF64[1, 0], 0.0, times; eom=:unsupported)
    end

    @testset "basis operator lifecycle applies operators across sectors" begin
        source = SpinBasis1D(4; nup=2, pauli=false, kblock=0)
        target = SpinBasis1D(4; nup=2, pauli=false, kblock=1)
        operator = [
            ("z", [1], 1.2),
            ("z", [2], -0.4),
            ("z", [4], 0.7im),
        ]
        vectors = reshape(
            ComplexF64.(1:(2length(source))),
            length(source),
            2,
        )
        full = SpinBasis1D(4; pauli=false)
        full_operator = sum(
            operator_matrix(full, opstring, [(coupling, sites...)])
            for (opstring, sites, coupling) in operator
        )
        expected =
            projection_matrix(target, ComplexF64; sparse=true)' *
            full_operator *
            projection_matrix(source, ComplexF64; sparse=true) *
            vectors
        shifted = op_shift_sector(target, source, operator, vectors)
        @test shifted ≈ expected atol=2e-14
        out = similar(shifted)
        @test op_shift_sector(target, source, operator, vectors; out) === out
        @test out ≈ expected atol=2e-14

        basis = SpinBasis1D(3; pauli=false)
        ket_states = UInt64[0, 1, 2, 7]
        matrix_elements, bra_states, surviving_kets = op_bra_ket(
            basis,
            "+",
            [2],
            1.3,
            ComplexF64,
            ket_states,
        )
        @test matrix_elements == ComplexF64[1.3, 1.3]
        @test bra_states == UInt64[2, 3]
        @test surviving_kets == UInt64[0, 1]

        full_elements, full_bras, full_kets = op_bra_ket(
            basis,
            "+",
            [2],
            1.3,
            ComplexF64,
            ket_states;
            reduce_output=false,
        )
        @test full_elements == ComplexF64[1.3, 1.3, 0, 0]
        @test full_bras == UInt64[2, 3, 2, 7]
        @test full_kets == ket_states
    end

    @testset "general boson, fermion, and user bases shift sectors" begin
        boson_source = BosonBasisGeneral(3; Nb=1, sps=3)
        boson_target = BosonBasisGeneral(3; Nb=2, sps=3)
        boson_state = ComplexF64[0.3, -0.2im, 0.7]
        boson_shifted = op_shift_sector(
            boson_target,
            boson_source,
            [("+", [2], 0.6)],
            boson_state,
        )
        boson_full = BosonBasis1D(3; sps=3)
        boson_reference =
            projection_matrix(
                boson_target,
                ComplexF64;
                sparse=true,
            )' *
            operator_matrix(boson_full, "+", [(0.6, 2)]) *
            projection_matrix(
                boson_source,
                ComplexF64;
                sparse=true,
            ) *
            boson_state
        @test boson_shifted ≈ boson_reference atol=3e-14

        fermion_source = SpinlessFermionBasisGeneral(3; Nf=1)
        fermion_target = SpinlessFermionBasisGeneral(3; Nf=2)
        fermion_state = ComplexF64[0.2, 0.4im, -0.7]
        fermion_shifted = op_shift_sector(
            fermion_target,
            fermion_source,
            [("+", [2], 1.1)],
            fermion_state,
        )
        fermion_full = SpinlessFermionBasis1D(3)
        fermion_reference =
            projection_matrix(
                fermion_target,
                ComplexF64;
                sparse=true,
            )' *
            operator_matrix(fermion_full, "+", [(1.1, 2)]) *
            projection_matrix(
                fermion_source,
                ComplexF64;
                sparse=true,
            ) *
            fermion_state
        @test fermion_shifted ≈ fermion_reference atol=3e-14

        flip(state, site) =
            (xor(state, UInt64(1) << (site - 1)), 1.0)
        user_source = UserBasis(
            UInt64,
            2,
            Dict("x" => flip);
            states=UInt64[0, 2],
        )
        user_target = UserBasis(
            UInt64,
            2,
            Dict("x" => flip);
            states=UInt64[1, 3],
        )
        user_state = ComplexF64[0.4, -0.7im]
        @test op_shift_sector(
            user_target,
            user_source,
            [("x", [1], 0.8)],
            user_state,
        ) ≈ 0.8user_state atol=2e-16

        elements, bras, kets = op_bra_ket(
            boson_source,
            "+",
            [2],
            0.6,
            ComplexF64,
            UInt64[0, 3],
        )
        @test elements ≈ ComplexF64[0.6, 0.6sqrt(2)]
        @test bras == UInt64[3, 6]
        @test kets == UInt64[0, 3]
    end

    @testset "general spin basis supports independent finite-order maps" begin
        Lx, Ly = 2, 3
        site(x, y) = x + Lx * y
        translation_x = [
            site(mod(x + 1, Lx), y)
            for y in 0:(Ly - 1) for x in 0:(Lx - 1)
        ]
        translation_y = [
            site(x, mod(y + 1, Ly))
            for y in 0:(Ly - 1) for x in 0:(Lx - 1)
        ]
        @test SpinBasisGeneral !== SpinBasis1D
        basis = SpinBasisGeneral(
            Lx * Ly;
            Nup=3,
            pauli=false,
            kxblock=(translation_x, 1),
            kyblock=(translation_y, 2),
            block_order=[:kxblock, :kyblock],
        )
        projector = projection_matrix(basis, ComplexF64; sparse=true)
        @test projector' * projector ≈
            Matrix{ComplexF64}(I, length(basis), length(basis)) atol=3e-14
        @test basis.blocks[:kxblock] == 1
        @test basis.blocks[:kyblock] == 2

        full = SpinBasis1D(Lx * Ly; nup=3, pauli=false)
        function permutation_matrix(map)
            rows = Int[]
            columns = Int[]
            for (column, encoded) in pairs(full.states)
                transformed = zero(UInt64)
                for source in 1:length(map)
                    iszero(encoded & (UInt64(1) << (source - 1))) && continue
                    transformed |= UInt64(1) << map[source]
                end
                push!(rows, state_index(full, transformed))
                push!(columns, column)
            end
            return sparse(
                rows,
                columns,
                ones(ComplexF64, length(rows)),
                length(full),
                length(full),
            )
        end
        Tx = permutation_matrix(translation_x)
        Ty = permutation_matrix(translation_y)
        parent_projector = basis.symmetry.projector
        @test parent_projector' * Tx * parent_projector ≈
            cis(2π / Lx) .* I atol=4e-14
        @test parent_projector' * Ty * parent_projector ≈
            cis(2π * 2 / Ly) .* I atol=4e-14

        couplings = [(0.4, 1), (-0.7, 4)]
        @test operator_matrix(basis, "z", couplings) ≈
            parent_projector' *
            operator_matrix(full, "z", couplings) *
            parent_projector atol=4e-14
    end

    @testset "general boson and fermion maps preserve statistics" begin
        Lx, Ly = 2, 2
        site(x, y) = x + Lx * y
        translation_x = [
            site(mod(x + 1, Lx), y)
            for y in 0:(Ly - 1) for x in 0:(Lx - 1)
        ]
        translation_y = [
            site(x, mod(y + 1, Ly))
            for y in 0:(Ly - 1) for x in 0:(Lx - 1)
        ]
        bosons = BosonBasisGeneral(
            4;
            Nb=2,
            sps=3,
            kxblock=(translation_x, 0),
            kyblock=(translation_y, 1),
        )
        fermions = SpinlessFermionBasisGeneral(
            4;
            Nf=2,
            kxblock=(translation_x, 1),
            kyblock=(translation_y, 0),
        )
        for basis in (bosons, fermions)
            P = basis.symmetry.projector
            @test P' * P ≈
                Matrix{ComplexF64}(I, length(basis), length(basis)) atol=4e-14
            @test basis.blocks[:kxblock_period] == 2
            @test basis.blocks[:kyblock_period] == 2
            parent = typeof(basis).parameters[1] === :boson ?
                BosonBasis1D(4; Nb=2, sps=3) :
                SpinlessFermionBasis1D(4; Nf=2)
            couplings = [(0.3, 1), (-0.2, 4)]
            @test operator_matrix(basis, "n", couplings) ≈
                P' * operator_matrix(parent, "n", couplings) * P atol=5e-14
        end

        no_doubles = SpinfulFermionBasisGeneral(
            3;
            Nf=(1, 1),
            double_occupancy=false,
        )
        @test length(no_doubles) == 6
        @test all(
            all(digit != 3 for digit in row)
            for row in eachrow(no_doubles.occupations)
        )

        particle_hole_even = SpinlessFermionBasisGeneral(
            2;
            Nf=1,
            phblock=([-1, -2], 0),
        )
        particle_hole_odd = SpinlessFermionBasisGeneral(
            2;
            Nf=1,
            phblock=([-1, -2], 1),
        )
        @test length(particle_hole_even) == 1
        @test length(particle_hole_odd) == 1
        @test abs.(
            particle_hole_even.symmetry.projector[:, 1],
        ) ≈ fill(inv(sqrt(2)), 2) atol=4e-14
        @test particle_hole_even.symmetry.projector' *
            particle_hole_odd.symmetry.projector ≈ zeros(1, 1) atol=4e-14

        spinful_particle_hole = SpinfulFermionBasisGeneral(
            1;
            Nf=[(0, 0), (1, 1)],
            simple_symm=false,
            phblock=([-1, -2], 0),
        )
        @test length(spinful_particle_hole) == 1
        @test abs.(
            spinful_particle_hole.symmetry.projector[:, 1],
        ) ≈ fill(inv(sqrt(2)), 2) atol=4e-14
    end

    @testset "UserBasis executes callback symmetries and local operators" begin
        N = 4
        rotate(state, N, args) = begin
            mask = (UInt64(1) << N) - 1
            ((UInt64(state) << 1) & mask) | (UInt64(state) >> (N - 1))
        end
        X = ComplexF64[0 1; 1 0]
        Z = ComplexF64[1 0; 0 -1]
        basis = UserBasis(
            UInt64,
            N,
            Dict("x" => X, "z" => Z);
            allowed_ops=("x", "z"),
            parallel=true,
            translation=(rotate, 4, 1, ()),
            block_order=[:translation],
        )
        P = basis.base.symmetry.projector
        @test length(basis) == size(P, 2)
        @test P' * P ≈
            Matrix{ComplexF64}(I, length(basis), length(basis)) atol=4e-14
        @test basis.blocks[:translation] == 1
        @test basis.blocks[:translation_period] == 4
        @test basis.blocks[:parallel]

        parent = UserBasis(
            UInt64,
            N,
            Dict("x" => X, "z" => Z);
            allowed_ops=("x", "z"),
        )
        couplings = [(0.3, 1), (-0.5, 3)]
        @test operator_matrix(basis, "z", couplings) ≈
            P' * operator_matrix(parent, "z", couplings) * P atol=4e-14

        constrained(state) =
            count_ones(state) == 3 &&
            iszero(state & (state << 1))
        sequential = UserBasis(
            UInt64,
            10,
            Dict("z" => Z);
            pre_check_state=constrained,
            parallel=false,
        )
        threaded = UserBasis(
            UInt64,
            10,
            Dict("z" => Z);
            pre_check_state=constrained,
            parallel=true,
        )
        @test states(threaded) == states(sequential)
    end

    @testset "higher-spin bases use exact angular-momentum matrix elements" begin
        spin_one = SpinBasis1D(1; S="1", pauli=false)
        @test spin_one.sps == 3
        @test spin_one.Ns == 3
        @test operator_matrix(spin_one, "z", [(1.0, 1)]) ==
            ComplexF64[-1 0 0; 0 0 0; 0 0 1]
        @test operator_matrix(spin_one, "+", [(1.0, 1)]) ≈
            ComplexF64[0 0 0; sqrt(2) 0 0; 0 sqrt(2) 0] atol=2e-15
        @test operator_matrix(spin_one, "-", [(1.0, 1)]) ≈
            ComplexF64[0 sqrt(2) 0; 0 0 sqrt(2); 0 0 0] atol=2e-15
        @test operator_matrix(spin_one, "x", [(1.0, 1)]) ≈
            ComplexF64[0 inv(sqrt(2)) 0; inv(sqrt(2)) 0 inv(sqrt(2)); 0 inv(sqrt(2)) 0] atol=2e-15
        @test operator_matrix(spin_one, "y", [(1.0, 1)]) ≈
            ComplexF64[0 im / sqrt(2) 0; -im / sqrt(2) 0 im / sqrt(2); 0 -im / sqrt(2) 0] atol=2e-15

        sector = SpinBasis1D(2; S=1, Nup=2, pauli=false)
        @test length(sector) == 3
        @test all(sum(row) == 2 for row in eachrow(sector.occupations))
        H = Hamiltonian(
            sector,
            [
                OperatorTerm("+-", [(0.7, 1, 2)]),
                OperatorTerm("-+", [(0.7, 1, 2)]),
                OperatorTerm("zz", [(0.2, 1, 2)]),
            ],
        )
        @test ishermitian(H)
        @test sort(eigvals(H)) ≈
            sort(eigvals(Hermitian(Matrix(H)))) atol=2e-14
    end

    @testset "multi-factor tensor operators and selective traces" begin
        first_spin = SpinBasis1D(1; pauli=false)
        boson = BosonBasis1D(1; sps=2)
        last_spin = SpinBasis1D(1; pauli=false)
        basis = TensorBasis(first_spin, boson, last_spin)
        @test basis.N == (1, 1, 1)
        @test state_index(basis, 2, 1, 2) == 6
        operator = operator_matrix(
            basis,
            "z|n|x",
            [(0.7, 1, 1, 1)],
        )
        expected = kron(
            operator_matrix(first_spin, "z", [(0.7, 1)]),
            operator_matrix(boson, "n", [(1.0, 1)]),
            operator_matrix(last_spin, "x", [(1.0, 1)]),
        )
        @test operator ≈ expected atol=2e-15
        static = Any[Any["x||y", [(0.6, 1, 1)]]]
        expanded_static, _ = expanded_form(basis, static, Any[])
        @test operator_matrix(
            basis,
            static[1][1],
            static[1][2],
        ) ≈ sum(
            operator_matrix(
                basis,
                entry[1],
                entry[2],
            )
            for entry in expanded_static
        ) atol=4e-15

        state = zeros(ComplexF64, length(basis))
        state[state_index(basis, 1, 1, 1)] = inv(sqrt(2))
        state[state_index(basis, 2, 2, 2)] = inv(sqrt(2))
        rho_outer, rho_middle = partial_trace(
            basis,
            state;
            sub_sys_A=[1, 3],
            return_rdm=:both,
        )
        @test size(rho_outer) == (4, 4)
        @test size(rho_middle) == (2, 2)
        @test tr(rho_outer) ≈ 1
        @test tr(rho_middle) ≈ 1
        @test sort(real.(eigvals(Hermitian(rho_middle)))) ≈ [0.5, 0.5]
        product = zeros(ComplexF64, length(basis))
        product[state_index(basis, 1, 1, 1)] = 1
        batch_entropy = ent_entropy(
            basis,
            hcat(state, product);
            sub_sys_A=[1, 3],
            enforce_pure=true,
            density=false,
            return_rdm=:A,
        )
        @test batch_entropy["Sent_A"] ≈ [log(2), 0.0] atol=4e-14
        @test size(batch_entropy["rdm_A"]) == (4, 4, 2)
    end

    @testset "PhotonBasis Ntot selects correlated particle-photon states" begin
        basis = PhotonBasis(SpinBasis1D, 3; Ntot=2, pauli=false)
        @test basis.Ntot == 2
        @test basis.Ns == photon_Hspace_dim(3, 2, nothing)
        @test basis.Ns == 7
        @test basis.blocks[:Ntot] == 2

        number_total = operator_matrix(
            basis,
            "z|",
            [(1.0, site) for site in 1:3],
        ) + operator_matrix(basis, "|n", [(1.0, 1)])
        @test number_total ≈
            (-3 / 2 + 2) .* Matrix{ComplexF64}(I, length(basis), length(basis)) atol=3e-15

        exchange = operator_matrix(
            basis,
            "+|-",
            [(1.0, 1, 1)],
        )
        @test !iszero(norm(exchange))
        @test size(exchange) == (length(basis), length(basis))
    end

    @testset "spinful sectors, Majorana operators, and species trace" begin
        multi = SpinfulFermionBasis1D(
            3;
            Nf=[(1, 0), (0, 1)],
        )
        @test length(multi) == 6
        @test all(
            begin
                up = count(digit -> digit & 1 == 1, row)
                down = count(digit -> digit & 2 == 2, row)
                (up, down) in ((1, 0), (0, 1))
            end
            for row in eachrow(multi.occupations)
        )

        spinless = SpinlessFermionBasis1D(1)
        majorana_x = operator_matrix(spinless, "x", [(1.0, 1)])
        majorana_y = operator_matrix(spinless, "y", [(1.0, 1)])
        @test majorana_x^2 ≈ I
        @test majorana_y^2 ≈ I
        @test majorana_x * majorana_y + majorana_y * majorana_x ≈ zeros(2, 2)

        spinful = SpinfulFermionBasis1D(2)
        up_x = operator_matrix(spinful, "x|", [(1.0, 1)])
        down_y = operator_matrix(spinful, "|y", [(1.0, 2)])
        @test up_x' ≈ up_x
        @test down_y' ≈ down_y

        state = zeros(ComplexF64, length(spinful))
        state[state_index(spinful, 9)] = inv(sqrt(2))
        state[state_index(spinful, 6)] = inv(sqrt(2))
        rho_A, rho_B = partial_trace(
            spinful,
            state;
            sub_sys_A=([1], [2]),
            return_rdm=:both,
        )
        @test size(rho_A) == (4, 4)
        @test tr(rho_A) ≈ 1
        @test tr(rho_B) ≈ 1

        spin_swap = [3, 4, 1, 2]
        swapped = SpinfulFermionBasisGeneral(
            2;
            Nf=(1, 1),
            simple_symm=false,
            swapblock=(spin_swap, 0),
        )
        @test swapped.blocks[:swapblock_period] == 2
        @test swapped.symmetry.projector' * swapped.symmetry.projector ≈
            Matrix{ComplexF64}(I, length(swapped), length(swapped)) atol=4e-14
    end

    @testset "dynamic symmetry blocks preserve drives and evolution" begin
        blocks = [Dict(:nup => sector) for sector in 0:2]
        static = [OperatorTerm("z", [(0.3, 1), (-0.2, 2)])]
        drive(time, omega) = cos(omega * time)
        dynamic = Any[Any[
            operator_matrix(
                SpinBasis1D(2; pauli=false),
                "zz",
                [(0.7, 1, 2)],
            ),
            drive,
            (1.4,),
        ]]
        projector, blocked = block_diag_hamiltonian(
            blocks,
            static,
            dynamic,
            SpinBasis1D,
            (2,),
            ComplexF64;
            basis_kwargs=Dict(:pauli => false),
        )
        full_basis = SpinBasis1D(2; pauli=false)
        full = Hamiltonian(
            static,
            dynamic;
            basis=full_basis,
            dtype=ComplexF64,
        )
        for time in (0.0, 0.3, 0.9)
            @test toarray(blocked; time) ≈
                projector' * toarray(full; time) * projector atol=4e-14
        end

        operator = BlockOps(
            blocks,
            static,
            dynamic,
            SpinBasis1D,
            (2,),
            ComplexF64;
            basis_kwargs=Dict(:pauli => false),
            compute_all_blocks=true,
        )
        initial = normalize(ComplexF64[1, 2im, -1, 0.5])
        times = [0.0, 0.2, 0.6]
        @test evolve(
            operator,
            initial,
            0.0,
            times;
            max_step=0.02,
            rtol=1e-10,
            atol=1e-12,
        ) ≈ evolve(
            full,
            initial,
            0.0,
            times;
            max_step=0.02,
            rtol=1e-10,
            atol=1e-12,
        ) atol=5e-10
        @test evolve(
            operator,
            initial,
            0.0,
            times;
            stack_state=true,
            max_step=0.02,
            rtol=1e-10,
            atol=1e-12,
        ) ≈ evolve(
            full,
            initial,
            0.0,
            times;
            max_step=0.02,
            rtol=1e-10,
            atol=1e-12,
        ) atol=5e-10
    end

    @testset "expanded operator forms preserve spin and fermion matrices" begin
        drive(time, frequency) = sin(frequency * time)
        for basis in (
            SpinBasis1D(2; pauli=false),
            SpinBasis1D(2; pauli=true),
            SpinBasisGeneral(1; S=1),
            SpinlessFermionBasisGeneral(2),
        )
            static = Any[Any["x", [(0.7, 1)]]]
            dynamic = Any[Any["y", [(-0.4, 1)], drive, (1.3,)]]
            expanded_static, expanded_dynamic =
                expanded_form(basis, static, dynamic)
            @test all(
                !occursin('x', String(entry[1])) &&
                !occursin('y', String(entry[1]))
                for entry in expanded_static
            )
            @test all(
                !occursin('x', String(entry[1])) &&
                !occursin('y', String(entry[1]))
                for entry in expanded_dynamic
            )
            original = Hamiltonian(
                static,
                dynamic;
                basis,
                dtype=ComplexF64,
                check_herm=false,
                check_pcon=false,
                check_symm=false,
            )
            expanded = Hamiltonian(
                expanded_static,
                expanded_dynamic;
                basis,
                dtype=ComplexF64,
                check_herm=false,
                check_pcon=false,
                check_symm=false,
            )
            for time in (0.0, 0.21, 0.8)
                @test toarray(expanded; time) ≈
                    toarray(original; time) atol=4e-14
            end
        end
    end

    @testset "batched entropy and harmonic-oscillator basis" begin
        basis = SpinBasis1D(2; pauli=false)
        bell = ComplexF64[inv(sqrt(2)), 0, 0, inv(sqrt(2))]
        product = ComplexF64[1, 0, 0, 0]
        batch = hcat(bell, product)
        entropy = ent_entropy(
            basis,
            batch;
            sub_sys_A=[1],
            density=false,
            enforce_pure=true,
            return_rdm=:both,
            return_rdm_EVs=true,
        )
        @test entropy["Sent_A"] ≈ [log(2), 0.0] atol=4e-14
        @test entropy["Sent_B"] ≈ entropy["Sent_A"] atol=4e-14
        @test size(entropy["rdm_A"]) == (2, 2, 2)
        @test size(entropy["rdm_B"]) == (2, 2, 2)
        @test size(entropy["p_A"]) == (2, 2)
        legacy = ent_entropy(
            Dict("V_states" => batch),
            basis;
            chain_subsys=[1],
            density=false,
        )
        @test legacy["Sent"] ≈ [log(2), 0.0] atol=4e-14

        mixed = cat(
            bell * bell',
            Diagonal(ComplexF64[0.5, 0, 0, 0.5]);
            dims=3,
        )
        measured = obs_vs_time(
            mixed,
            [0.0, 0.5],
            Dict("identity" => Matrix{ComplexF64}(I, 4, 4));
            Sent_args=Dict(
                :basis => basis,
                :sub_sys_A => [1],
                :density => false,
            ),
        )
        @test measured["Sent_time"]["Sent_A"] ≈
            [log(2), log(2)] atol=4e-14

        oscillator = HOBasis(3)
        @test isbasis(oscillator)
        @test oscillator.Np == 3
        @test oscillator.N == 1
        @test length(oscillator) == 4
        raising = operator_matrix(
            oscillator,
            "+",
            [(1.0, 1)];
            sparse=false,
        )
        expected = zeros(4, 4)
        expected[2, 1] = 1
        expected[3, 2] = sqrt(2)
        expected[4, 3] = sqrt(3)
        @test raising ≈ expected atol=4e-14
        @test length(HOBasis(0)) == 1
    end

    @testset "generic evolution stacks complex states and accepts scalar time" begin
        rotation(time, state, frequency) = begin
            n = length(state) ÷ 2
            return vcat(
                frequency .* @view(state[(n + 1):end]),
                -frequency .* @view(state[1:n]),
            )
        end
        initial = ComplexF64[1 + 2im, -0.5 + 0.25im]
        frequency = 1.7
        times = [0.0, 0.2, 0.7]
        stacked = evolve(
            initial,
            0.0,
            times,
            rotation;
            stack_state=true,
            f_params=(frequency,),
            solver_name=:zvode,
            max_step=0.01,
            rtol=1e-10,
            atol=1e-12,
        )
        @test stacked ≈
            initial .* transpose(exp.(-im .* frequency .* times)) atol=2e-9
        scalar = evolve(
            initial,
            0.0,
            0.3,
            rotation;
            stack_state=true,
            f_params=(frequency,),
            solver_name=:dop853,
            max_step=0.01,
        )
        @test scalar ≈ initial .* exp(-im * frequency * 0.3) atol=2e-9
        @test scalar isa Vector

        basis = SpinBasis1D(1; pauli=false)
        drive(time, omega) = sin(omega * time)
        H = Hamiltonian(
            Any[Any["z", [(0.3, 1)]]],
            Any[Any["x", [(0.4, 1)], drive, (1.1,)]];
            basis,
            dtype=ComplexF64,
        )
        state = ComplexF64[1, im] ./ sqrt(2)
        grid = evolve(H, state, 0.0, [0.0, 0.3]; max_step=0.01)
        @test evolve(H, state, 0.0, 0.3; max_step=0.01) ≈
            grid[:, 2] atol=3e-10
    end

    @testset "dynamic Hamiltonian matrix arithmetic and powers" begin
        basis = SpinBasis1D(1; pauli=false)
        drive(time, frequency) = cos(frequency * time)
        H = Hamiltonian(
            Any[Any["z", [(0.4, 1)]]],
            Any[Any["x", [(0.7, 1)], drive, (1.2,)]];
            basis,
            dtype=ComplexF64,
        )
        matrix = ComplexF64[0.2 0.1im; -0.1im -0.3]
        for time in (0.0, 0.23, 0.9)
            dense = toarray(H; time)
            @test toarray(H + matrix; time) ≈ dense + matrix atol=4e-14
            @test toarray(matrix - H; time) ≈ matrix - dense atol=4e-14
            @test toarray(H * matrix; time) ≈ dense * matrix atol=4e-14
            @test toarray(matrix * H; time) ≈ matrix * dense atol=4e-14
            @test toarray(H^3; time) ≈ dense^3 atol=7e-14
        end
    end

    @testset "QuantumOperator defaults and symbolic arithmetic" begin
        basis = SpinBasis1D(1; pauli=false)
        X = operator_matrix(basis, "x", [(1.0, 1)])
        Z = operator_matrix(basis, "z", [(1.0, 1)])
        left = QuantumOperator(basis, Dict(:x => X, :z => Z))
        right = QuantumOperator(
            basis,
            Dict(:z => 0.2Z, :identity => Matrix{ComplexF64}(I, 2, 2)),
        )
        @test toarray(left) ≈ X + Z atol=2e-16
        @test toarray(left; pars=Dict(:x => 2.0)) ≈ 2X + Z atol=2e-16
        @test_throws ArgumentError toarray(
            left;
            pars=Dict(:unknown => 1.0),
        )
        parameters = Dict(:x => 0.7, :z => -0.3, :identity => 0.4)
        @test toarray(left + right; pars=parameters) ≈
            0.7X - 0.3(1.2Z) +
            0.4Matrix{ComplexF64}(I, 2, 2) atol=3e-16
        @test toarray(left - right; pars=parameters) ≈
            0.7X - 0.3(0.8Z) -
            0.4Matrix{ComplexF64}(I, 2, 2) atol=3e-16
        @test toarray(-2left; pars=Dict(:x => 0.7, :z => -0.3)) ≈
            -2 .* (0.7X - 0.3Z) atol=3e-16
        @test toarray(left / 4; pars=Dict(:x => 0.7, :z => -0.3)) ≈
            (0.7X - 0.3Z) ./ 4 atol=3e-16
    end

    @testset "project_from sparse output controls" begin
        spin = SpinBasis1D(4; nup=2, kblock=0)
        vector = ComplexF64.(1:length(spin))
        vectors = hcat(vector, 2vector)
        sparse_vector = sparse(vector)
        sparse_vectors = sparse(vectors)

        @test project_from(spin, vector; sparse=true) isa SparseVector
        @test project_from(spin, vectors; sparse=true) isa SparseMatrixCSC
        @test project_from(spin, sparse_vector; sparse=false) isa Vector
        @test project_from(spin, sparse_vectors; sparse=false) isa Matrix
        @test project_from(spin, vector; sparse=true) ≈
            project_from(spin, vector; sparse=false)

        boson = BosonBasis1D(3; Nb=2, sps=3, kblock=0)
        boson_state = ComplexF64.(1:length(boson))
        @test project_from(boson, boson_state; sparse=true) isa SparseVector
        @test project_from(boson, sparse(boson_state); sparse=false) isa Vector

        tensor = TensorBasis(SpinBasis1D(2; nup=1), HOBasis(2))
        tensor_state = ComplexF64.(1:length(tensor))
        @test project_from(tensor, tensor_state; sparse=true) isa SparseVector
        @test project_from(tensor, tensor_state; sparse=false) isa Vector

        parent_state = project_from(
            spin,
            vector;
            sparse=false,
            pcon=true,
        )
        @test length(parent_state) == length(spin.symmetry.parent_states)
        @test parent_state ≈ spin.symmetry.projector * vector atol=3e-14
        @test projection_matrix(
            spin,
            ComplexF64;
            sparse=true,
            pcon=true,
        ) ≈ spin.symmetry.projector atol=3e-14
        @test project_to(
            spin,
            parent_state;
            sparse=false,
            pcon=true,
        ) ≈ vector atol=3e-14
        @test project_to(
            spin,
            project_from(spin, vector; sparse=false);
            sparse=false,
        ) ≈ vector atol=3e-14
        @test project_from(
            spin,
            vector;
            sparse=true,
            pcon=true,
        ) isa SparseVector

        boson_parent = project_from(
            boson,
            boson_state;
            sparse=false,
            pcon=true,
        )
        @test boson_parent ≈
            boson.symmetry.projector * boson_state atol=3e-14

        tensor_left = SpinBasis1D(4; nup=2, kblock=0)
        tensor_right = BosonBasis1D(3; Nb=1, sps=2, kblock=0)
        pcon_tensor = TensorBasis(tensor_left, tensor_right)
        pcon_tensor_state = ComplexF64.(1:length(pcon_tensor))
        tensor_parent = project_from(
            pcon_tensor,
            pcon_tensor_state;
            sparse=false,
            pcon=true,
        )
        @test tensor_parent ≈
            kron(
                tensor_left.symmetry.projector,
                tensor_right.symmetry.projector,
            ) * pcon_tensor_state atol=4e-14
        @test project_to(
            pcon_tensor,
            tensor_parent;
            sparse=false,
            pcon=true,
        ) ≈ pcon_tensor_state atol=4e-14
    end

    @testset "general bases support deferred construction" begin
        translation = [1, 2, 3, 0]
        deferred = SpinBasisGeneral(
            4;
            Nup=2,
            pauli=false,
            tblock=(translation, 0),
            make_basis=false,
        )
        @test length(deferred) == 1
        @test deferred.blocks[:made_basis] == false
        @test_throws ArgumentError operator_matrix(
            deferred,
            "z",
            [(1.0, 1)],
        )
        @test_throws ArgumentError make_basis_blocks(deferred)
        @test make_basis!(deferred) === deferred
        eager = SpinBasisGeneral(
            4;
            Nup=2,
            pauli=false,
            tblock=(translation, 0),
        )
        @test deferred.blocks[:made_basis] == true
        @test states(deferred) == states(eager)
        @test projection_matrix(deferred, ComplexF64; sparse=true) ≈
            projection_matrix(eager, ComplexF64; sparse=true)

        deferred_boson = BosonBasisGeneral(
            3;
            Nb=2,
            sps=3,
            tblock=([1, 2, 0], 0),
            make_basis=false,
        )
        @test length(deferred_boson) == 1
        make_basis!(deferred_boson)
        eager_boson = BosonBasisGeneral(
            3;
            Nb=2,
            sps=3,
            tblock=([1, 2, 0], 0),
        )
        @test states(deferred_boson) == states(eager_boson)
        @test operator_matrix(
            deferred_boson,
            "n",
            [(0.7, 2)],
        ) ≈ operator_matrix(eager_boson, "n", [(0.7, 2)])

        deferred_user = UserBasis(
            UInt64,
            4,
            Dict("z" => ComplexF64[1 0; 0 -1]);
            pre_check_state=state -> iseven(count_ones(state)),
            _make_basis=false,
        )
        @test length(deferred_user) == 1
        @test deferred_user.blocks[:made_basis] == false
        make_basis!(deferred_user)
        @test states(deferred_user) == UInt64[0, 3, 5, 6, 9, 10, 12, 15]
        @test deferred_user.blocks[:made_basis] == true
    end

    @testset "general spin bases use wide encoded integers" begin
        basis = SpinBasisGeneral(70; Nup=1, pauli=false)
        @test basis.dtype === UInt256
        @test length(basis) == 70
        @test state_at(basis, 70) == UInt256(BigInt(1) << 69)
        encoded = int_to_state(basis, state_at(basis, 70))
        @test state_to_int(basis, encoded) == state_at(basis, 70)

        hopping = operator_matrix(basis, "+-", [(0.7, 70, 1)])
        @test count(!iszero, hopping) == 1
        @test hopping[state_index(basis, BigInt(1) << 69), 1] ≈ 0.7
        @test_throws ArgumentError projection_matrix(basis; sparse=true)

        translation = vcat(collect(1:69), 0)
        momentum = SpinBasisGeneral(
            70;
            Nup=1,
            pauli=false,
            kblock=(translation, 3),
        )
        @test momentum.dtype === UInt256
        @test length(momentum) == 1
        @test operator_matrix(
            momentum,
            "z",
            [(1.0, 70)],
        )[1, 1] ≈ -34 / 70 atol=3e-14
    end
end
