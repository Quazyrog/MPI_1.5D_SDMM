#ifndef DISTRIBUTED_SPARSE_MATRIX_HPP
#define DISTRIBUTED_SPARSE_MATRIX_HPP
#include <vector>
#include <cstddef>
#include <iostream>
#include <spdlog/spdlog.h>
#include "Commons.hpp"

class SparseMatrix
{
    size_t nrows_, ncols_;
    std::vector<size_t> rows_offsets_;
    std::vector<size_t> columns_indices_;
    std::vector<double> data_;

public:

    static SparseMatrix Read(std::istream &stream);
};


class CSRReadError: public IOError
{
public:
    template<class ...Args>
    explicit CSRReadError(std::string_view format, Args... args):
        IOError("CSRReadError", format, std::forward<Args>(args)...)
    {}
};

#endif // DISTRIBUTED_SPARSE_MATRIX_HPP
