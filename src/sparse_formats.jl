# Define multiplication methods for these types individually. Julia 1.10's
# LinearAlgebra uses several narrower structured-matrix unions, so one combined
# RHS union would leave method intersections ambiguous.
const _STRUCTURED_MATRIX_RHS_TYPES = (
    Bidiagonal,
    Diagonal,
    SymTridiagonal,
    Tridiagonal,
    LinearAlgebra.AbstractTriangular,
)

"""
    SparseMatrixCSR(matrix)

Minimal Julia-native compressed-sparse-row matrix used to preserve QuSpin's
public `csr` storage contract. Julia's standard library is CSC-first, so this
adapter deliberately implements only the matrix protocol needed by QuSpin:
indexing, matrix-vector/matrix multiplication, conversion, transpose, and
adjoint. It never labels CSC data as CSR.
"""
struct SparseMatrixCSR{Tv,Ti<:Integer} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
end

function SparseMatrixCSR(matrix::SparseMatrixCSC{T,Ti}) where {T,Ti}
    m, n = size(matrix)
    rowptr = zeros(Int, m + 1)
    rowptr[1] = 1
    @inbounds for row in matrix.rowval
        rowptr[row + 1] += 1
    end
    cumsum!(rowptr, rowptr)
    next = copy(rowptr)
    colval = Vector{Int}(undef, nnz(matrix))
    nzval = Vector{T}(undef, nnz(matrix))
    @inbounds for column in 1:n
        for pointer in matrix.colptr[column]:(matrix.colptr[column + 1] - 1)
            row = matrix.rowval[pointer]
            destination = next[row]
            colval[destination] = column
            nzval[destination] = matrix.nzval[pointer]
            next[row] += 1
        end
    end
    return SparseMatrixCSR{T,Int}(m, n, rowptr, colval, nzval)
end

SparseMatrixCSR(matrix::AbstractMatrix) = SparseMatrixCSR(sparse(matrix))

Base.size(matrix::SparseMatrixCSR) = (matrix.m, matrix.n)
Base.eltype(::Type{SparseMatrixCSR{Tv,Ti}}) where {Tv,Ti} = Tv
Base.eltype(matrix::SparseMatrixCSR) = eltype(typeof(matrix))
Base.copy(matrix::SparseMatrixCSR) = SparseMatrixCSR(
    matrix.m,
    matrix.n,
    copy(matrix.rowptr),
    copy(matrix.colval),
    copy(matrix.nzval),
)

function Base.getindex(matrix::SparseMatrixCSR{T}, row::Int, column::Int) where {T}
    checkbounds(matrix, row, column)
    for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
        stored_column = matrix.colval[pointer]
        stored_column == column && return matrix.nzval[pointer]
        stored_column > column && break
    end
    return zero(T)
end

function Base.Matrix(matrix::SparseMatrixCSR{T}) where {T}
    result = zeros(T, size(matrix))
    for row in 1:matrix.m
        for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
            result[row, matrix.colval[pointer]] += matrix.nzval[pointer]
        end
    end
    return result
end

function SparseArrays.sparse(matrix::SparseMatrixCSR{T}) where {T}
    colptr = zeros(Int, matrix.n + 1)
    colptr[1] = 1
    @inbounds for column in matrix.colval
        colptr[column + 1] += 1
    end
    cumsum!(colptr, colptr)
    next = copy(colptr)
    rowval = Vector{Int}(undef, length(matrix.nzval))
    nzval = Vector{T}(undef, length(matrix.nzval))
    @inbounds for row in 1:matrix.m
        for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
            column = matrix.colval[pointer]
            destination = next[column]
            rowval[destination] = row
            nzval[destination] = matrix.nzval[pointer]
            next[column] += 1
        end
    end
    return SparseMatrixCSC(matrix.m, matrix.n, colptr, rowval, nzval)
end

SparseArrays.nnz(matrix::SparseMatrixCSR) = length(matrix.nzval)
SparseArrays.SparseMatrixCSC(matrix::SparseMatrixCSR) = sparse(matrix)

function _csr_mul!(
    result::AbstractVecOrMat,
    matrix::SparseMatrixCSR,
    value::AbstractVecOrMat,
    alpha::Number,
    beta::Number,
)
    size(matrix, 2) == size(value, 1) ||
        throw(DimensionMismatch("matrix and value dimensions do not match"))
    expected_size = value isa AbstractVector ?
        (size(matrix, 1),) :
        (size(matrix, 1), size(value, 2))
    size(result) == expected_size ||
        throw(DimensionMismatch("output has the wrong dimensions"))
    input = Base.mightalias(result, value) ? copy(value) : value
    iszero(beta) ? fill!(result, zero(eltype(result))) : lmul!(beta, result)
    if value isa AbstractVector
        for row in 1:matrix.m
            total = zero(promote_type(eltype(matrix), eltype(input)))
            for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
                total += matrix.nzval[pointer] * input[matrix.colval[pointer]]
            end
            result[row] += alpha * total
        end
    else
        for row in 1:matrix.m
            for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
                column = matrix.colval[pointer]
                coefficient = alpha * matrix.nzval[pointer]
                @inbounds for rhs in axes(input, 2)
                    result[row, rhs] += coefficient * input[column, rhs]
                end
            end
        end
    end
    return result
end

function _csr_mul(matrix::SparseMatrixCSR, value::AbstractVector)
    T = promote_type(eltype(matrix), eltype(value))
    result = zeros(T, size(matrix, 1))
    for row in 1:matrix.m
        total = zero(T)
        for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
            total += matrix.nzval[pointer] * value[matrix.colval[pointer]]
        end
        result[row] = total
    end
    return result
end

function _csr_mul(matrix::SparseMatrixCSR, value::AbstractMatrix)
    T = promote_type(eltype(matrix), eltype(value))
    result = zeros(T, size(matrix, 1), size(value, 2))
    for row in 1:matrix.m
        for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
            column = matrix.colval[pointer]
            coefficient = matrix.nzval[pointer]
            @inbounds for rhs in axes(value, 2)
                result[row, rhs] += coefficient * value[column, rhs]
            end
        end
    end
    return result
end

Base.:*(matrix::SparseMatrixCSR{T}, value::AbstractVector{S}) where {T,S} =
    _csr_mul(matrix, value)
Base.:*(matrix::SparseMatrixCSR{T}, value::AbstractMatrix{S}) where {T,S} =
    _csr_mul(matrix, value)
LinearAlgebra.mul!(
    result::AbstractVector,
    matrix::SparseMatrixCSR,
    value::AbstractVector,
) = _csr_mul!(result, matrix, value, true, false)
LinearAlgebra.mul!(
    result::AbstractMatrix,
    matrix::SparseMatrixCSR,
    value::AbstractMatrix,
) = _csr_mul!(result, matrix, value, true, false)
LinearAlgebra.mul!(
    result::AbstractVector,
    matrix::SparseMatrixCSR,
    value::AbstractVector,
    alpha::Number,
    beta::Number,
) = _csr_mul!(result, matrix, value, alpha, beta)
LinearAlgebra.mul!(
    result::AbstractMatrix,
    matrix::SparseMatrixCSR,
    value::AbstractMatrix,
    alpha::Number,
    beta::Number,
) = _csr_mul!(result, matrix, value, alpha, beta)

for StructuredMatrixRHS in _STRUCTURED_MATRIX_RHS_TYPES
    @eval Base.:*(
        matrix::SparseMatrixCSR,
        value::$StructuredMatrixRHS,
    ) = _csr_mul(matrix, value)
    @eval LinearAlgebra.mul!(
        result::AbstractMatrix,
        matrix::SparseMatrixCSR,
        value::$StructuredMatrixRHS,
    ) = _csr_mul!(result, matrix, value, true, false)
    @eval LinearAlgebra.mul!(
        result::AbstractMatrix,
        matrix::SparseMatrixCSR,
        value::$StructuredMatrixRHS,
        alpha::Number,
        beta::Number,
    ) = _csr_mul!(result, matrix, value, alpha, beta)
end

function _csr_transpose(matrix::SparseMatrixCSR{T}; conjugate::Bool=false) where {T}
    rowptr = zeros(Int, matrix.n + 1)
    rowptr[1] = 1
    @inbounds for column in matrix.colval
        rowptr[column + 1] += 1
    end
    cumsum!(rowptr, rowptr)
    next = copy(rowptr)
    colval = Vector{Int}(undef, length(matrix.nzval))
    nzval = Vector{T}(undef, length(matrix.nzval))
    @inbounds for row in 1:matrix.m
        for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
            new_row = matrix.colval[pointer]
            destination = next[new_row]
            colval[destination] = row
            nzval[destination] =
                conjugate ? conj(matrix.nzval[pointer]) : matrix.nzval[pointer]
            next[new_row] += 1
        end
    end
    return SparseMatrixCSR{T,Int}(matrix.n, matrix.m, rowptr, colval, nzval)
end

Base.transpose(matrix::SparseMatrixCSR) = _csr_transpose(matrix)
Base.adjoint(matrix::SparseMatrixCSR) = _csr_transpose(matrix; conjugate=true)
Base.conj(matrix::SparseMatrixCSR{T}) where {T} = SparseMatrixCSR{T,Int}(
    matrix.m,
    matrix.n,
    copy(matrix.rowptr),
    copy(matrix.colval),
    conj.(matrix.nzval),
)
LinearAlgebra.ishermitian(matrix::SparseMatrixCSR) =
    size(matrix, 1) == size(matrix, 2) && sparse(matrix) == adjoint(sparse(matrix))

"""
    DIAMatrix(matrix)

Sparse diagonal storage with one compact vector per populated diagonal. This
is the native counterpart of SciPy's DIA format and is primarily useful for
banded operators and storage-format compatibility.
"""
struct DIAMatrix{Tv,Ti<:Integer} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    offsets::Vector{Ti}
    diagonals::Vector{Vector{Tv}}
end

function DIAMatrix(matrix::AbstractMatrix{T}) where {T}
    csc = sparse(matrix)
    rows, columns, values = findnz(csc)
    grouped = Dict{Int,Vector{Tuple{Int,T}}}()
    for (row, column, value) in zip(rows, columns, values)
        offset = column - row
        push!(get!(grouped, offset, Tuple{Int,T}[]), (row, value))
    end
    offsets = sort!(collect(keys(grouped)))
    diagonals = Vector{Vector{T}}()
    for offset in offsets
        first_row = max(1, 1 - offset)
        last_row = min(size(matrix, 1), size(matrix, 2) - offset)
        values_on_diagonal = zeros(T, max(0, last_row - first_row + 1))
        for (row, value) in grouped[offset]
            values_on_diagonal[row - first_row + 1] += value
        end
        push!(diagonals, values_on_diagonal)
    end
    return DIAMatrix{T,Int}(
        size(matrix, 1),
        size(matrix, 2),
        offsets,
        diagonals,
    )
end

Base.size(matrix::DIAMatrix) = (matrix.m, matrix.n)
Base.eltype(::Type{DIAMatrix{Tv,Ti}}) where {Tv,Ti} = Tv
Base.eltype(matrix::DIAMatrix) = eltype(typeof(matrix))
Base.copy(matrix::DIAMatrix) = DIAMatrix(
    matrix.m,
    matrix.n,
    copy(matrix.offsets),
    copy.(matrix.diagonals),
)

function Base.getindex(matrix::DIAMatrix{T}, row::Int, column::Int) where {T}
    checkbounds(matrix, row, column)
    offset = column - row
    position = searchsortedfirst(matrix.offsets, offset)
    position <= length(matrix.offsets) &&
        matrix.offsets[position] == offset ||
        return zero(T)
    first_row = max(1, 1 - offset)
    return matrix.diagonals[position][row - first_row + 1]
end

function Base.Matrix(matrix::DIAMatrix{T}) where {T}
    result = zeros(T, size(matrix))
    for (offset, diagonal) in zip(matrix.offsets, matrix.diagonals)
        first_row = max(1, 1 - offset)
        for (index, value) in pairs(diagonal)
            row = first_row + index - 1
            result[row, row + offset] = value
        end
    end
    return result
end

function SparseArrays.sparse(matrix::DIAMatrix{T}) where {T}
    rows = Int[]
    columns = Int[]
    values = T[]
    for (offset, diagonal) in zip(matrix.offsets, matrix.diagonals)
        first_row = max(1, 1 - offset)
        for (index, value) in pairs(diagonal)
            iszero(value) && continue
            row = first_row + index - 1
            push!(rows, row)
            push!(columns, row + offset)
            push!(values, value)
        end
    end
    return sparse(rows, columns, values, matrix.m, matrix.n)
end

SparseArrays.nnz(matrix::DIAMatrix) =
    sum(count(!iszero, diagonal) for diagonal in matrix.diagonals)
SparseArrays.SparseMatrixCSC(matrix::DIAMatrix) = sparse(matrix)

function _dia_mul!(
    result::AbstractVecOrMat,
    matrix::DIAMatrix,
    value::AbstractVecOrMat,
    alpha::Number,
    beta::Number,
)
    size(matrix, 2) == size(value, 1) ||
        throw(DimensionMismatch("matrix and value dimensions do not match"))
    expected_size = value isa AbstractVector ?
        (size(matrix, 1),) :
        (size(matrix, 1), size(value, 2))
    size(result) == expected_size ||
        throw(DimensionMismatch("output has the wrong dimensions"))
    input = Base.mightalias(result, value) ? copy(value) : value
    iszero(beta) ? fill!(result, zero(eltype(result))) : lmul!(beta, result)
    for (offset, diagonal) in zip(matrix.offsets, matrix.diagonals)
        first_row = max(1, 1 - offset)
        for (index, coefficient) in pairs(diagonal)
            iszero(coefficient) && continue
            row = first_row + index - 1
            column = row + offset
            scaled_coefficient = alpha * coefficient
            if input isa AbstractVector
                result[row] += scaled_coefficient * input[column]
            else
                @inbounds for rhs in axes(input, 2)
                    result[row, rhs] += scaled_coefficient * input[column, rhs]
                end
            end
        end
    end
    return result
end

function _dia_mul(matrix::DIAMatrix, value::AbstractVector)
    T = promote_type(eltype(matrix), eltype(value))
    result = zeros(T, size(matrix, 1))
    for (offset, diagonal) in zip(matrix.offsets, matrix.diagonals)
        first_row = max(1, 1 - offset)
        for (index, coefficient) in pairs(diagonal)
            iszero(coefficient) && continue
            row = first_row + index - 1
            result[row] += coefficient * value[row + offset]
        end
    end
    return result
end

function _dia_mul(matrix::DIAMatrix, value::AbstractMatrix)
    T = promote_type(eltype(matrix), eltype(value))
    result = zeros(T, size(matrix, 1), size(value, 2))
    for (offset, diagonal) in zip(matrix.offsets, matrix.diagonals)
        first_row = max(1, 1 - offset)
        for (index, coefficient) in pairs(diagonal)
            iszero(coefficient) && continue
            row = first_row + index - 1
            column = row + offset
            @inbounds for rhs in axes(value, 2)
                result[row, rhs] += coefficient * value[column, rhs]
            end
        end
    end
    return result
end

Base.:*(matrix::DIAMatrix{T}, value::AbstractVector{S}) where {T,S} =
    _dia_mul(matrix, value)
Base.:*(matrix::DIAMatrix{T}, value::AbstractMatrix{S}) where {T,S} =
    _dia_mul(matrix, value)
LinearAlgebra.mul!(
    result::AbstractVector,
    matrix::DIAMatrix,
    value::AbstractVector,
) = _dia_mul!(result, matrix, value, true, false)
LinearAlgebra.mul!(
    result::AbstractMatrix,
    matrix::DIAMatrix,
    value::AbstractMatrix,
) = _dia_mul!(result, matrix, value, true, false)
LinearAlgebra.mul!(
    result::AbstractVector,
    matrix::DIAMatrix,
    value::AbstractVector,
    alpha::Number,
    beta::Number,
) = _dia_mul!(result, matrix, value, alpha, beta)
LinearAlgebra.mul!(
    result::AbstractMatrix,
    matrix::DIAMatrix,
    value::AbstractMatrix,
    alpha::Number,
    beta::Number,
) = _dia_mul!(result, matrix, value, alpha, beta)

for StructuredMatrixRHS in _STRUCTURED_MATRIX_RHS_TYPES
    @eval Base.:*(
        matrix::DIAMatrix,
        value::$StructuredMatrixRHS,
    ) = _dia_mul(matrix, value)
    @eval LinearAlgebra.mul!(
        result::AbstractMatrix,
        matrix::DIAMatrix,
        value::$StructuredMatrixRHS,
    ) = _dia_mul!(result, matrix, value, true, false)
    @eval LinearAlgebra.mul!(
        result::AbstractMatrix,
        matrix::DIAMatrix,
        value::$StructuredMatrixRHS,
        alpha::Number,
        beta::Number,
    ) = _dia_mul!(result, matrix, value, alpha, beta)
end

function _dia_transpose(matrix::DIAMatrix{T}; conjugate::Bool=false) where {T}
    offsets = Int[-offset for offset in Iterators.reverse(matrix.offsets)]
    diagonals = Vector{Vector{T}}(undef, length(matrix.diagonals))
    for (destination, source) in enumerate(Iterators.reverse(matrix.diagonals))
        diagonals[destination] = conjugate ? conj.(source) : copy(source)
    end
    return DIAMatrix{T,Int}(matrix.n, matrix.m, offsets, diagonals)
end

Base.transpose(matrix::DIAMatrix) = _dia_transpose(matrix)
Base.adjoint(matrix::DIAMatrix) = _dia_transpose(matrix; conjugate=true)
Base.conj(matrix::DIAMatrix{T}) where {T} = DIAMatrix{T,Int}(
    matrix.m,
    matrix.n,
    copy(matrix.offsets),
    conj.(matrix.diagonals),
)
LinearAlgebra.ishermitian(matrix::DIAMatrix) =
    size(matrix, 1) == size(matrix, 2) && sparse(matrix) == adjoint(sparse(matrix))
