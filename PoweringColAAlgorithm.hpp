#ifndef POWERING_COL_A_ALGORITHM_HPP
#define POWERING_COL_A_ALGORITHM_HPP

#include <mpi.h>
#include "MatrixPoweringAlgorithm.hpp"


struct ColASettings
{
    int dense_matrix_seed;
    int c_param;
};


class PoweringColAAlgorithm: public MatrixPoweringAlgorithm
{
    ColASettings settings_;
    std::shared_ptr<SparseMatrixSplitter> splitter_;
    long problem_size_ = -1;
    SparseMatrixData a_;
    DenseMatrix b_;
    DenseMatrix c_;

    MPI_Comm world2d_;
    int world2d_my_coords_[2];
    int world2d_my_rank_;
    int world2d_ring_prev_;
    int world2d_ring_next_;
    int world2d_ring_coordinator_;

public:
    ~PoweringColAAlgorithm() override;

    explicit PoweringColAAlgorithm(const ColASettings &settings);
    std::shared_ptr<SparseMatrixSplitter> init_splitter(long sparse_rows, long sparse_columns) override;
    void initialize(SparseMatrixData &&sparse_part) override;
    void replicate() override;
};

#endif // POWERING_COL_A_ALGORITHM_HPP
