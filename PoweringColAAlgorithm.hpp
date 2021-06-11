#ifndef POWERING_COL_A_ALGORITHM_HPP
#define POWERING_COL_A_ALGORITHM_HPP
#include "MatrixPoweringAlgorithm.hpp"


struct ColASettings
{
    int dense_matrix_seed;
};


class PoweringColAAlgorithm: public MatrixPoweringAlgorithm
{
    ColASettings settings_;
    std::shared_ptr<SparseMatrixSplitter> splitter_;
    long problem_size_ = -1;
    SparseMatrixData a_;
    DenseMatrix b_;

    void generate_dense_part_();

public:

    explicit PoweringColAAlgorithm(const ColASettings &settings);
    std::shared_ptr<SparseMatrixSplitter> init_splitter(long sparse_rows, long sparse_columns) override;
    void initialize(SparseMatrixData &&sparse_part) override;
};

#endif // POWERING_COL_A_ALGORITHM_HPP
