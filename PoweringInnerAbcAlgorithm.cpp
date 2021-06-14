#include "PoweringInnerAbcAlgorithm.hpp"
#include "densematgen.h"
#include "Commons.hpp"
#include "Debug.hpp"

namespace {
class Splitter: public SparseMatrixSplitter
{
    std::optional<DataDistribution1D> rows_distrib_;
    std::vector<SparseEntry> entries_;

public:
    void assign_matrix_data(long rows, long columns, std::vector<SparseEntry> &&data) override
    {
        rows_distrib_ = DataDistribution1D(static_cast<int>(rows), static_cast<int>(NumberOfProcesses));
        entries_.swap(data);
        std::sort(entries_.begin(), entries_.end(), CSROrder{});
    }

    std::pair<SparseEntry *, size_t> range_of(int process_rank) override
    {
        const auto rows_begin = static_cast<long>(rows_distrib_->offset(static_cast<size_t>(process_rank)));
        auto first_elt = std::lower_bound(entries_.begin(), entries_.end(), SparseEntry{rows_begin, -1, 0}, CSROrder{});
        const auto rows_end = static_cast<long>(rows_distrib_->offset(static_cast<size_t>(process_rank) + 1));
        auto last_elt = std::lower_bound(entries_.begin(), entries_.end(), SparseEntry{rows_end, -1, 0}, CSROrder{});
        return {&(*first_elt), last_elt - first_elt};
    }

    void free() override
    {
        std::vector<SparseEntry>{}.swap(entries_);
    }
};
}


PoweringInnerABCAlgorithm::PoweringInnerABCAlgorithm(const int c_param)
{
    // Initialize coords_to_rank_
    layer_size_ = c_param;
    for (int r = 0; r < NumberOfProcesses; ++r) {
        const int j = coord_ring_of_(r);
        const int l = coord_layer_of_(r);
        coords_to_rank_[{j, l}] = r;
    }

    // Initialize communication data
    ring_size_ = NumberOfProcesses / c_param;
    coord_ring_ = coord_ring_of_(ProcessRank);
    coord_layer_ = coord_layer_of_(ProcessRank);
    number_of_phases_ = NumberOfProcesses / (c_param * c_param);
    ring_prev_rank_ = coords_to_rank_.at({(coord_ring_ + 1) % ring_size_, coord_layer_});
    ring_next_rank_ = coords_to_rank_.at({(coord_ring_ + ring_size_ - 1) % ring_size_, coord_layer_});

    // Split communicators
    int new_rank;
    MPI_Comm_split(MPI_COMM_WORLD, coord_layer_, coord_ring_, &ring_comm_);
    MPI_Comm_rank(ring_comm_, &new_rank);
    assert(new_rank == coord_ring_);
    MPI_Comm_split(MPI_COMM_WORLD, coord_ring_, coord_layer_, &layer_comm_);
    MPI_Comm_rank(layer_comm_, &new_rank);
    assert(new_rank == coord_layer_);

    spdlog::info("Cartesian coordinates: (ring={}, layer={}); in the ring prev={}, next={}", coord_ring_, coord_layer_,
                 ring_prev_rank_, ring_next_rank_);
}

PoweringInnerABCAlgorithm::~PoweringInnerABCAlgorithm()
{
    MPI_Comm_free(&ring_comm_);
    MPI_Comm_free(&layer_comm_);
}

std::shared_ptr<SparseMatrixSplitter> PoweringInnerABCAlgorithm::init_splitter(long sparse_rows, long sparse_columns)
{
    if (sparse_rows != sparse_columns)
        throw ValueError("Matrix A was supposed to be square, not {}x{}", sparse_rows, sparse_columns);
    return std::make_shared<Splitter>();
}

void PoweringInnerABCAlgorithm::initialize(SparseMatrixData &&sparse_part, const int dense_seed)
{
    std::swap(a_, sparse_part);
    problem_size_ = a_.columns;
    spdlog::debug("Problem size is {}", problem_size_);

    DataDistribution1D dist(problem_size_, NumberOfProcesses);
    const auto nrows = static_cast<long>(problem_size_);
    const auto first_column = dist.offset(ProcessRank);
    const auto ncols = static_cast<long>(dist.offset(ProcessRank + 1) - first_column);
    ColumnMajorMatrix b(nrows, ncols);
    b.in_order_foreach([first_column, dense_seed](auto r, auto c, auto &v) {
        v = generate_double(dense_seed, r, first_column + c);
    });
    std::swap(b_, b);

    seed_ = dense_seed;
}

void PoweringInnerABCAlgorithm::replicate()
{
    replicate_a_(layer_comm_, layer_size_, coord_ring_);
    replicate_b_(layer_comm_);
    init_inbox_(ring_comm_);
    initial_shift_();
}

void PoweringInnerABCAlgorithm::replicate_b_(MPI_Comm &layer)
{
    // We can actually determine these things locally :)
    DataDistribution1D cols_dist(problem_size_, static_cast<size_t>(NumberOfProcesses));
    int total_size = 0;
    std::vector<int> sizes;
    sizes.reserve(layer_size_);
    std::vector<int> offsets;
    sizes.reserve(layer_size_);

    // Compute sizes and offsets
    const int first_in_layer = ProcessRank - coord_layer_;
    const int first_out_layer = first_in_layer + layer_size_;
    for (int proc = first_in_layer; proc <= first_out_layer; ++proc) {
        const int size = static_cast<int>(cols_dist.offset(proc + 1) - cols_dist.offset(proc))
            * static_cast<int>(problem_size_);
        sizes.push_back(size);
        offsets.push_back(total_size);
        total_size += size;
    }
    spdlog::debug("Gathered B parts sizes in layer {}: {}", coord_ring_, Debug::VectorToString(sizes));

    // Gather data of B
    spdlog::debug("Gathering {} elements of dense matrix on layer {} (columns {} : {})", total_size, coord_ring_,
                  cols_dist.offset(first_in_layer), cols_dist.offset(first_out_layer));
    std::vector<double> gathered_data(total_size);
    MPI_Allgatherv(b_.data(), static_cast<int>(b_.size()), MPI_DOUBLE,
                   gathered_data.data(), sizes.data(), offsets.data(), MPI_DOUBLE, layer);

    // Combine received data into B (it is column major order, so it works)
    const int n_cols = static_cast<int>(cols_dist.offset(first_out_layer) - cols_dist.offset(first_in_layer));
    b_ = ColumnMajorMatrix(static_cast<int>(problem_size_), n_cols, std::move(gathered_data));
    c_ = ColumnMajorMatrix(static_cast<int>(problem_size_), n_cols);

    // Check the data
    if constexpr(Debug::ENABLED) {
        const auto first_column = cols_dist.offset(first_in_layer);
        b_.in_order_foreach([this, first_column](auto r, auto c, auto v) {
            c += first_column;
            const auto expected = generate_double(seed_, r, c);
            if (v != expected) {
                spdlog::critical("Unexpected value after gather: B[{}, {}]={} != {}", r, c, v, expected);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        });
        spdlog::debug("Gathered B passed the check");
    }
}

void PoweringInnerABCAlgorithm::initial_shift_()
{
    const int my_shift = number_of_phases_ * coord_layer_;
    if (my_shift == 0) {
        spdlog::info("Initial shift by 0");
        return;
    }

    const auto prev_rank = coords_to_rank_.at({(coord_ring_ + ring_size_ - my_shift % ring_size_) % ring_size_, coord_layer_});
    const auto next_rank = coords_to_rank_.at({(coord_ring_ + my_shift) % ring_size_, coord_layer_});
    std::array<MPI_Request, 6> requests;
    rotate_a_(requests.data(), next_rank, prev_rank);
    spdlog::info("Initial shift by {}: receive from {}, send to {}", my_shift, prev_rank, next_rank);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    std::swap(a_, inbox_);
}

void PoweringInnerABCAlgorithm::multiply()
{
    c_.in_order_foreach([](auto, auto, auto &v) {
        v = 0;
    });

    for (int phase = 0; phase < number_of_phases_; ++phase) {
        spdlog::info("Multiplication phase {} of {} started", phase + 1, number_of_phases_);
        std::array<MPI_Request, 6> requests;
        rotate_a_(requests.data(), ring_next_rank_, ring_prev_rank_);

        SparseDenseMultiply(a_, b_, c_);

        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        std::swap(a_, inbox_);
    }

    MPI_Allreduce(MPI_IN_PLACE, c_.data(), static_cast<int>(c_.size()), MPI_DOUBLE, MPI_SUM, layer_comm_);
}

void PoweringInnerABCAlgorithm::swap_cb()
{
    std::swap(b_, c_);
}

std::optional<ColumnMajorMatrix> PoweringInnerABCAlgorithm::gather_result()
{
    std::vector<double> result_data;
    std::vector<int> sizes, offsets;
    if (ProcessRank == COORDINATOR_WORLD_RANK) {
        result_data.resize(problem_size_ * problem_size_);

        // Now we need to compute sizes for all layers
        sizes.resize(ring_size_, 0);
        DataDistribution1D dist(problem_size_, NumberOfProcesses);
        for (int proc = 0; proc < NumberOfProcesses; ++proc) {
            const auto proc_layer = coord_ring_of_(proc);
            sizes[proc_layer] += static_cast<int>(dist.offset(proc + 1) - dist.offset(proc));
        }
        for (auto &size: sizes)
            size *= static_cast<int>(problem_size_);

        // And from them we compute offsets
        int total_size = 0;
        offsets.reserve(sizes.size());
        for (auto &size: sizes) {
            offsets.push_back(total_size);
            total_size += size;
        }
        assert(total_size == result_data.size());
        spdlog::trace("Gather sizes are: {}", Debug::VectorToString(sizes));
    }

    if (coord_layer_ == coord_layer_of_(COORDINATOR_WORLD_RANK)) {
        spdlog::debug("Gathering result from {} to {}", coord_ring_, coord_ring_of_(COORDINATOR_WORLD_RANK));
        MPI_Gatherv(c_.data(), static_cast<int>(c_.size()), MPI_DOUBLE,
                    result_data.data(), sizes.data(), offsets.data(), MPI_DOUBLE,
                    coord_ring_of_(COORDINATOR_WORLD_RANK), ring_comm_);
        spdlog::debug("Done!");
    }

    if (ProcessRank == COORDINATOR_WORLD_RANK)
        return ColumnMajorMatrix(static_cast<long>(problem_size_), static_cast<long>(problem_size_), std::move(result_data));
    return std::optional<ColumnMajorMatrix>();
}
