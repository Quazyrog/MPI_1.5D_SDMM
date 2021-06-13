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
    for (int r = 0; r < NumberOfProcesses; ++r) {
        const int j = r / c_param;
        const int l = r % c_param;
        coords_to_rank_[{j, l}] = r;
    }

    // Initialize communication data
    layer_size_ = c_param;
    ring_size_ = NumberOfProcesses / c_param;
    coord_ring_ = ProcessRank / c_param;
    coord_layer_ = ProcessRank % c_param;
    number_of_phases_ = NumberOfProcesses / (c_param * c_param);
    ring_prev_rank_ = coords_to_rank_.at({(coord_ring_ + 1) % ring_size_, coord_layer_});
    ring_next_rank_ = coords_to_rank_.at({(coord_ring_ + ring_size_ - 1) % ring_size_, coord_layer_});
    MPI_Comm_split(MPI_COMM_WORLD, coord_layer_, ProcessRank, &layer_comm_);

    spdlog::info("Cartesian coordinates: (ring={}, layer={}); in the ring prev={}, next={}", coord_ring_, coord_layer_,
                 ring_prev_rank_, ring_next_rank_);
}

PoweringInnerABCAlgorithm::~PoweringInnerABCAlgorithm()
{
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
    MPI_Comm layer;
    MPI_Comm_split(MPI_COMM_WORLD, coord_ring_, ProcessRank, &layer);
    replicate_a_(layer);
    replicate_b_(layer);
    MPI_Comm_free(&layer);
}

void PoweringInnerABCAlgorithm::replicate_a_(MPI_Comm &layer)
{
    // Gather sizes
    std::vector<int> sizes(layer_size_);
    int size = static_cast<int>(a_.size());
    MPI_Allgather(&size, 1, MPI_INT, sizes.data(), 1, MPI_INT, layer);

    // Prepare offsets for Allgatherv and compute combined size
    std::vector<int> offsets;
    offsets.reserve(layer_size_);
    int total_size = 0;
    for (auto s: sizes) {
        offsets.push_back(total_size);
        total_size += s;
    }

    // Serialize to triples
    std::vector<SparseEntry> local_entries;
    local_entries.reserve(a_.size());
    a_.in_order_foreach_nonzero([&local_entries](auto r, auto c, auto v) {
        local_entries.push_back(SparseEntry{r, c, v});
    });
    assert(a_.size() == local_entries.size());

    // Gather them all
    spdlog::trace("Layer {} offsets array: {}", coord_ring_, Debug::VectorToString(offsets));
    MPI_Datatype sparse_entry_datatype;
    SparseEntry::InitMPIDataType(sparse_entry_datatype);
    std::vector<SparseEntry> gathered_entries(total_size);
    MPI_Allgatherv(local_entries.data(), static_cast<int>(local_entries.size()), sparse_entry_datatype,
                   gathered_entries.data(), sizes.data(), offsets.data(), sparse_entry_datatype, layer);

    // Reconstruct new matrix
    spdlog::info("Layer {} has {} entries in its sparse matrix", coord_ring_, total_size);
    spdlog::trace("Gathered entries: {}", Debug::VectorToString(gathered_entries));
    a_ = SparseMatrixData::BuildCSR(a_.rows, a_.columns, gathered_entries.size(), gathered_entries.data());
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
    const int first_in_layer = ProcessRank - (ProcessRank % layer_size_);
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
    spdlog::debug("Gathering {} elements of dense matrix on layer {} (columns {} through {})", total_size, coord_ring_,
                  cols_dist.offset(first_in_layer), cols_dist.offset(first_out_layer) - 1);
    std::vector<double> gathered_data(total_size);
    MPI_Allgatherv(b_.data(), static_cast<int>(b_.size()), MPI_DOUBLE,
                   gathered_data.data(), sizes.data(), offsets.data(), MPI_DOUBLE, layer);

    // Combine received data into B (it is column major order, so it works)
    const int n_cols = static_cast<int>(cols_dist.offset(first_out_layer) - cols_dist.offset(first_in_layer));
    b_ = ColumnMajorMatrix(static_cast<int>(problem_size_), n_cols, std::move(gathered_data));

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

void PoweringInnerABCAlgorithm::multiply()
{

}

void PoweringInnerABCAlgorithm::swap_cb()
{

}

std::optional<ColumnMajorMatrix> PoweringInnerABCAlgorithm::gather_result()
{
    return std::optional<ColumnMajorMatrix>();
}
