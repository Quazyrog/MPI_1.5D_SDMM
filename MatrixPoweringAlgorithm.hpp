#ifndef MATRIX_POWERING_ALGORITHM_HPP
#define MATRIX_POWERING_ALGORITHM_HPP
#include "Commons.hpp"
#include "Matrix.hpp"

class SparseMatrixSplitter
{
public:
    virtual void assign_matrix_data(long rows, long columns, std::vector<SparseEntry> &&data) = 0;
    virtual std::pair<SparseEntry *, size_t> range_of(int process_rank) = 0;
    virtual void free() = 0;
};

class MatrixPoweringAlgorithm
{
protected:
    SparseMatrixData inbox_, a_;
    void init_inbox_(MPI_Comm &ring);
    void replicate_a_(MPI_Comm &layer, int layer_size, int layer_num);
    void rotate_a_(MPI_Request *requests, int next, int prev);

public:
    virtual ~MatrixPoweringAlgorithm() = default;

    virtual std::shared_ptr<SparseMatrixSplitter> init_splitter(long sparse_rows, long sparse_columns) = 0;
    virtual void initialize(SparseMatrixData &&sparse_part, int dense_seed) = 0;
    virtual void replicate() = 0;

    virtual void multiply() = 0;
    virtual void swap_cb() = 0;

    virtual std::optional<ColumnMajorMatrix> gather_result() = 0;
};

#endif // MATRIX_POWERING_ALGORITHM_HPP
