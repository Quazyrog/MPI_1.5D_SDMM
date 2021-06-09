#include "SparseMatrix.hpp"

SparseMatrix SparseMatrix::Read(std::istream &stream)
{
    size_t nrows, ncols, entries, magic;
    if (!(stream >> nrows >> ncols >> entries >> magic))
        throw IOError("Failed to read header in sparse matrix file");
    if (magic != 4)
        spdlog::warn("CSR magic number: {} != 4", magic);
    spdlog::debug("Reading CSR matrix {}x{} with {} entries", nrows, ncols, entries);

    SparseMatrix result;
    result.nrows_ = nrows;
    result.ncols_ = ncols;
    result.rows_offsets_.resize(nrows + 1);
    result.columns_indices_.resize(entries);
    result.data_.resize(entries);

    // Read data
    for (size_t e = 0; e < entries; ++e) {
        double entry;
        if (!(stream >> entry))
            throw CSRReadError("Failed to read entry {}", entry);
        result.data_[e] = entry;
    }

    // Read rows offsets
    size_t last_offset = 0;
    for (int r = 0; r < nrows; ++r) {
        size_t offset;
        if (!(stream >> offset))
            throw CSRReadError("Failed to read row offset for row {}", r);
        if (offset > entries)
            throw CSRReadError("Row offset too large: {} for {} entries", offset, entries);
        if (last_offset > offset)
            throw CSRReadError("Non-monotonic row offsets: {} then {}", last_offset, offset);
        result.rows_offsets_[r] = offset;
    }
    size_t data_end_offset;
    if (!(stream >> data_end_offset))
        throw CSRReadError("Failed to read data end offset");
    if (data_end_offset != entries)
        spdlog::warn("Unexpected CSR data end offset: got {}, should be {}", data_end_offset, entries);
    result.rows_offsets_[nrows] = std::min(entries, std::max(last_offset, data_end_offset));

    // Read column indices
    for (size_t r = 0; r < nrows; ++r) {
        const auto row_begin = result.columns_indices_.begin() + static_cast<long>(result.rows_offsets_[r]);
        const auto row_end = result.columns_indices_.begin() + static_cast<long>(result.rows_offsets_[r + 1]);
        size_t last_index = 0;
        for (auto it = row_begin; it != row_end; ++it) {
            size_t index;
            if (!(stream >> index))
                throw CSRReadError("Failed to read column index for entry {}", it - result.columns_indices_.begin());
            if (last_index > index)
                throw CSRReadError("Non-increasing columns indices in row {}: {} then {}", r, last_index, index);
            *it = index;
        }
    }

    return result;
}
