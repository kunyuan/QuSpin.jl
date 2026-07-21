@testset "basis integer properties" begin
    @test QuSpin.Basis.UInt256 === UInt256
    @test QuSpin.Basis.UInt1024 === UInt1024
    @test QuSpin.Basis.UInt4096 === UInt4096
    @test QuSpin.Basis.UInt16384 === UInt16384

    python_oracles = [
        (0, UInt32),
        (BigInt(2)^32 - 1, UInt32),
        (BigInt(2)^32, UInt64),
        (BigInt(2)^64 - 1, UInt64),
        (BigInt(2)^64, UInt256),
        (BigInt(2)^255, UInt256),
        (BigInt(2)^256, UInt1024),
    ]
    for (value, expected_type) in python_oracles
        encoded = python_int_to_basis_int(value)
        @test encoded isa expected_type
        @test basis_int_to_python_int(encoded) == value
    end

    type_oracles = [
        ((10, 5, 2), UInt32),
        ((32, nothing, 2), UInt32),
        ((64, nothing, 2), UInt64),
        ((100, 1, 2), UInt256),
        ((300, nothing, 2), UInt1024),
        ((5000, nothing, 2), UInt16384),
    ]
    for ((N, Np, sps), expected_type) in type_oracles
        @test get_basis_type(N, Np, sps) === expected_type
    end

    for T in (UInt256, UInt1024, UInt4096, UInt16384)
        samples = (BigInt(0), BigInt(1), BigInt(7), (BigInt(1) << 63) + 9)
        @test collect(basis_int_to_python_int.(T.(samples))) == collect(samples)
        @test basis_zeros((2, 2), T) == fill(T(0), 2, 2)
        @test basis_ones((2, 2), T) == fill(T(1), 2, 2)
    end

    left = UInt32[0x0f, 0xf0, 0xaa]
    right = UInt32[0x33, 0x55, 0xff]
    @test bitwise_and(left, right) == UInt32[0x03, 0x50, 0xaa]
    @test bitwise_or(left, right) == UInt32[0x3f, 0xf5, 0xff]
    @test bitwise_xor(left, right) == UInt32[0x3c, 0xa5, 0x55]
    @test bitwise_not(UInt8[0x00, 0xff]) == UInt8[0xff, 0x00]
    @test bitwise_leftshift(UInt32[1, 2, 3], [1, 2, 3]) == UInt32[2, 8, 24]
    @test bitwise_rightshift(UInt32[8, 16, 32], [1, 2, 3]) == UInt32[4, 4, 4]

    # Algebraic properties exercise values beyond machine-word width.
    a = UInt256((BigInt(1) << 200) + 0x35)
    b = UInt256((BigInt(1) << 199) + 0x0f)
    @test bitwise_xor(bitwise_xor(a, b), b) == a
    @test bitwise_and(a, b) == bitwise_and(b, a)
    @test bitwise_or(a, b) == bitwise_or(b, a)
    @test bitwise_not(bitwise_not(a)) == a
end
