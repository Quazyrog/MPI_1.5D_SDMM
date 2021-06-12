#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
#include <cstddef>
#include <iostream>
#include <spdlog/spdlog.h>
#include "Commons.hpp"

struct SparseEntry
{
    long row, column;
    double value;

    static void InitMPIDataType(MPI_Datatype &type);
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

    template<class Function>
    void in_order_foreach_nonzero(const Function &function) {
        long row = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            while (i >= offsets[row + 1])
                ++row;
            function(row, indices[i], values[i]);
        }
    }

    static std::tuple<size_t, size_t, std::vector<SparseEntry>> ReadCSRFile(std::istream &stream);
    static SparseMatrixData BuildCSC(long rows, long cols, size_t count, SparseEntry *data);
    static SparseMatrixData BuildCSR(long rows, long cols, size_t count, SparseEntry *data);
};


class DenseMatrix
{
    long rows_ = 0, columns_ = 0;
    std::vector<double> values_;

public:
    DenseMatrix() = default;
    DenseMatrix(long rows, long columns):
        values_(rows * columns, 0)
    {}

    inline long rows() const noexcept
    { return rows_; }

    inline long columns() const noexcept
    { return columns_; }

    inline double &operator()(long row, long column) noexcept
    { return values_[column * rows_ + row]; }

    inline double operator()(long row, long column) const noexcept
    { return values_[column * rows_ + row]; }

    template<class Function>
    void in_order_foreach(Function body)
    {
        for (long col = 0; col < columns_; ++col) {
            for (long row = 0; row < columns_; ++row)
                body(row, col, values_[col * rows_ + row]);
        }
    }
};


class CSRReadError: public IOError
{
public:
    template<class ...Args>
    explicit CSRReadError(std::string_view format, Args... args):
        IOError("CSRReadError", format, std::forward<Args>(args)...)
    {}
};


void SparseDenseMultiply(const SparseMatrixData &csr_matrix, const DenseMatrix &dense_matrix,
                         const DenseMatrix &result_accumulator);

#endif // MATRIX_HPP
