#ifndef POWERING_INNER_ABC_ALGORITHM_HPP
#define POWERING_INNER_ABC_ALGORITHM_HPP
#include "Commons.hpp"
#include "MatrixPoweringAlgorithm.hpp"

/// InnerABC based algorithm as describen in Enlightnment.py :)
class PoweringInnerABCAlgorithm: public MatrixPoweringAlgorithm
{
    /// Maps (j, l) -> process_rank; j = coord_layer_, l = coord_ring_
    std::map<std::pair<int, int>, int> coords_to_rank_;
    int ring_size_;
    /// Number of this process in its ring (they rotate A)
    int coord_ring_;
    /// Number of this process in its layer (they replicate B and C)
    int coord_layer_;

    MPI_Comm layer_comm_;
    int ring_prev_rank_;
    int ring_next_rank_;

    int number_of_phases_;

    size_t problem_size_;
    SparseMatrixData a_;
    DenseMatrix b_;

public:
    explicit PoweringInnerABCAlgorithm(int c_param);
    ~PoweringInnerABCAlgorithm() override;

    std::shared_ptr<SparseMatrixSplitter> init_splitter(long sparse_rows, long sparse_columns) override;
    void initialize(SparseMatrixData &&sparse_part, int dense_seed) override;
    void replicate() override;
    void multiply() override;
    void swap_cb() override;
    std::optional<DenseMatrix> gather_result() override;
};

#endif // POWERING_INNER_ABC_ALGORITHM_HPP
