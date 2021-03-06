#include <iostream>
#include "Matrix.hpp"

std::tuple<long, long, std::vector<SparseEntry>> SparseMatrixData::ReadCSRFile(std::istream &stream)
{
    long nrows, ncols, entries, magic;
    if (!(stream >> nrows >> ncols >> entries >> magic))
        throw IOError("Failed to read header in sparse matrix file");
    if (magic != 4)
        spdlog::warn("CSR magic number: {} != 4", magic);
    spdlog::debug("Reading CSR matrix {}x{} with {} entries", nrows, ncols, entries);

    std::vector<SparseEntry> result(entries);

    // Read entries
    for (long e = 0; e < entries; ++e) {
        double entry;
        if (!(stream >> result[e].value))
            throw CSRReadError("Failed to read entry {}", entry);
    }

    // Read rows offsets
    long last_offset = 0;
    if (!(stream >> last_offset))
        throw CSRReadError("Failed to read entries end offset");
    if (last_offset != 0)
        throw CSRReadError("initial offset should be 0, got {}", last_offset);
    long entry = last_offset;
    for (int r = 0; r < nrows; ++r) {
        long offset;
        if (!(stream >> offset))
            throw CSRReadError("Failed to read row offset for row {}", r);
        if (offset > entries)
            throw CSRReadError("Row offset too large: {} for {} entries", offset, entries);
        for (; entry < offset; ++entry)
            result[entry].row = r;
        last_offset = offset;
    }
    if (last_offset != entries)
        spdlog::warn("Unexpected CSR entries end offset: got {}, should be {}", last_offset, entries);

    // Read column indices
    for (long e = 0; e < entries; ++e) {
        if (!(stream >> result[e].column))
            throw CSRReadError("Failed to read column index for entry {}", e);
    }

    return {nrows, ncols, result};
}


SparseMatrixData SparseMatrixData::BuildCSC(const long rows, const long cols, const size_t count, SparseEntry *data)
{
    SparseMatrixData matrix;
    matrix.rows = rows;
    matrix.columns = cols;

    std::sort(data, data + count, CSCOrder{});
    matrix.offsets.reserve(cols);
    matrix.indices.reserve(count);
    matrix.values.reserve(count);

    auto entry_index = 0;
    matrix.offsets.push_back(0);
    for (long c = 0; c < cols; ++c) {
        while (entry_index < count && data[entry_index].column == c) {
            matrix.indices.push_back(data[entry_index].row);
            matrix.values.push_back(data[entry_index].value);
            ++entry_index;
        }
        matrix.offsets.push_back(static_cast<long>(matrix.values.size()));
    }

    return matrix;
}


SparseMatrixData SparseMatrixData::BuildCSR(long rows, long cols, size_t count, SparseEntry *data)
{
    SparseMatrixData matrix;
    matrix.rows = rows;
    matrix.columns = cols;

    std::sort(data, data + count, CSROrder{});
    matrix.offsets.reserve(rows);
    matrix.indices.reserve(count);
    matrix.values.reserve(count);

    auto entry_index = 0;
    matrix.offsets.push_back(0);
    for (long r = 0; r < rows; ++r) {
        while (entry_index < count && data[entry_index].row == r) {
            matrix.indices.push_back(data[entry_index].column);
            matrix.values.push_back(data[entry_index].value);
            ++entry_index;
        }
        matrix.offsets.push_back(static_cast<long>(matrix.values.size()));
    }

    return matrix;
}

void SparseEntry::InitMPIDataType(MPI_Datatype &type)
{
    static const int block_lengths[3] = {1, 1, 1};
    static const MPI_Aint offsets[3] = {
        offsetof(SparseEntry, row),
        offsetof(SparseEntry, column),
        offsetof(SparseEntry, value)
    };
    MPI_Datatype types[3] = {MPI_LONG, MPI_LONG, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_lengths, offsets, types, &type);
    MPI_Type_commit(&type);
}

void SparseDenseMultiply(const SparseMatrixData &csr_matrix, const ColumnMajorMatrix &dense_matrix, ColumnMajorMatrix &out)
{
    for (long dense_column = 0; dense_column < dense_matrix.columns(); ++dense_column) {
        csr_matrix.in_order_foreach_nonzero([&out, &dense_matrix, dense_column](auto r, auto c, auto v) {
            assert(r < out.rows());
            assert(dense_column < out.columns());
            assert(c < dense_matrix.rows());
            assert(dense_column < dense_matrix.columns());
            out(r, dense_column) += v * dense_matrix(c, dense_column);
        });
    }
}

std::ostream &operator<<(std::ostream &s, const SparseEntry &e)
{
    return s << fmt::format("SparseEntry{{{}, {}, {}}}", e.row, e.column, e.value);
}
