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

function SparseMatrixCSR(matrix::AbstractMatrix{T}) where {T}
    csc = sparse(matrix)
    rows, columns, values = findnz(csc)
    order = sortperm(eachindex(values); by=index -> (rows[index], columns[index]))
    rowptr = zeros(Int, size(matrix, 1) + 1)
    rowptr[1] = 1
    for index in order
        rowptr[rows[index] + 1] += 1
    end
    cumsum!(rowptr, rowptr)
    return SparseMatrixCSR{T,Int}(
        size(matrix, 1),
        size(matrix, 2),
        rowptr,
        Int[columns[index] for index in order],
        T[values[index] for index in order],
    )
end

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
    rows = Int[]
    columns = Int[]
    values = T[]
    sizehint!(rows, length(matrix.nzval))
    sizehint!(columns, length(matrix.nzval))
    sizehint!(values, length(matrix.nzval))
    for row in 1:matrix.m
        for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
            push!(rows, row)
            push!(columns, matrix.colval[pointer])
            push!(values, matrix.nzval[pointer])
        end
    end
    return sparse(rows, columns, values, matrix.m, matrix.n)
end

SparseArrays.nnz(matrix::SparseMatrixCSR) = length(matrix.nzval)
SparseArrays.SparseMatrixCSC(matrix::SparseMatrixCSR) = sparse(matrix)

function _csr_mul(matrix::SparseMatrixCSR, value::AbstractVecOrMat)
    size(matrix, 2) == size(value, 1) ||
        throw(DimensionMismatch("matrix and value dimensions do not match"))
    T = promote_type(eltype(matrix), eltype(value))
    output_shape = value isa AbstractVector ?
        (size(matrix, 1),) :
        (size(matrix, 1), size(value, 2))
    result = zeros(T, output_shape)
    if value isa AbstractVector
        for row in 1:matrix.m
            total = zero(T)
            for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
                total += matrix.nzval[pointer] * value[matrix.colval[pointer]]
            end
            result[row] = total
        end
    else
        for row in 1:matrix.m
            for pointer in matrix.rowptr[row]:(matrix.rowptr[row + 1] - 1)
                column = matrix.colval[pointer]
                @views result[row, :] .+= matrix.nzval[pointer] .* value[column, :]
            end
        end
    end
    return result
end

Base.:*(matrix::SparseMatrixCSR{T}, value::AbstractVector{S}) where {T,S} =
    _csr_mul(matrix, value)
Base.:*(matrix::SparseMatrixCSR{T}, value::AbstractMatrix{S}) where {T,S} =
    _csr_mul(matrix, value)

Base.transpose(matrix::SparseMatrixCSR) = SparseMatrixCSR(transpose(sparse(matrix)))
Base.adjoint(matrix::SparseMatrixCSR) = SparseMatrixCSR(adjoint(sparse(matrix)))
Base.conj(matrix::SparseMatrixCSR) = SparseMatrixCSR(conj(sparse(matrix)))

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

function _dia_mul(matrix::DIAMatrix, value::AbstractVecOrMat)
    size(matrix, 2) == size(value, 1) ||
        throw(DimensionMismatch("matrix and value dimensions do not match"))
    T = promote_type(eltype(matrix), eltype(value))
    output_shape = value isa AbstractVector ?
        (size(matrix, 1),) :
        (size(matrix, 1), size(value, 2))
    result = zeros(T, output_shape)
    for (offset, diagonal) in zip(matrix.offsets, matrix.diagonals)
        first_row = max(1, 1 - offset)
        for (index, coefficient) in pairs(diagonal)
            iszero(coefficient) && continue
            row = first_row + index - 1
            column = row + offset
            if value isa AbstractVector
                result[row] += coefficient * value[column]
            else
                @views result[row, :] .+= coefficient .* value[column, :]
            end
        end
    end
    return result
end

Base.:*(matrix::DIAMatrix{T}, value::AbstractVector{S}) where {T,S} =
    _dia_mul(matrix, value)
Base.:*(matrix::DIAMatrix{T}, value::AbstractMatrix{S}) where {T,S} =
    _dia_mul(matrix, value)

Base.transpose(matrix::DIAMatrix) = DIAMatrix(transpose(sparse(matrix)))
Base.adjoint(matrix::DIAMatrix) = DIAMatrix(adjoint(sparse(matrix)))
Base.conj(matrix::DIAMatrix) = DIAMatrix(conj(sparse(matrix)))
