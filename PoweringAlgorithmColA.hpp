#ifndef POWERING_COL_A_ALGORITHM_HPP
#define POWERING_COL_A_ALGORITHM_HPP

#include <mpi.h>
#include "PoweringAlgorithm.hpp"


struct ColASettings
{
    int dense_matrix_seed;
    int c_param;
};


class PoweringAlgorithmColA: public PoweringAlgorithm
{
    ColASettings settings_;
    std::shared_ptr<SparseMatrixSplitter> splitter_;
    long problem_size_ = -1;
    ColumnMajorMatrix b_;
    ColumnMajorMatrix c_;

    int world2d_ring_prev_;
    int world2d_ring_next_;

public:
    explicit PoweringAlgorithmColA(const ColASettings &settings);
    std::shared_ptr<SparseMatrixSplitter> init_splitter(long sparse_rows, long sparse_columns) override;
    void initialize(SparseMatrixData &&sparse_part, int dense_seed) override;
    void replicate() override;

    void multiply() override;
    void swap_cb() override;

    std::optional<ColumnMajorMatrix> gather_result() override;
};

#endif // POWERING_COL_A_ALGORITHM_HPP
