#ifndef DISTRIBUTED_SPARSE_MATRIX_HPP
#define DISTRIBUTED_SPARSE_MATRIX_HPP
#include <vector>
#include <cstddef>
#include <iostream>
#include <spdlog/spdlog.h>
#include "Commons.hpp"

struct SparseEntry
{
    long row, column;
    double value;
};

struct CSROrder
{
    bool operator()(const SparseEntry &a, const SparseEntry &b)
    { return a.row < b.row || (a.row == b.row && a.column < b.column); }
};

struct CSCOrder
{
    bool operator()(const SparseEntry &a, const SparseEntry &b)
    { return a.column < b.column || (a.column == b.column && a.row < b.row); }
};


struct SparseMatrixData
{
    long rows = 0, columns = 0;
    std::vector<long> offsets;
    std::vector<long> indices;
    std::vector<double> values;

    static std::tuple<size_t, size_t, std::vector<SparseEntry>> ReadCSRFile(std::istream &stream);
    static SparseMatrixData BuildCSC(long rows, long cols, size_t count, SparseEntry *data);
    static SparseMatrixData BuildCSR(long rows, long cols, size_t count, SparseEntry *data);
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
