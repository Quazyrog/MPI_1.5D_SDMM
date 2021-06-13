#include "densematgen.h"
#include "PoweringColAAlgorithm.hpp"


namespace {
class Splitter: public SparseMatrixSplitter
{
    long rows_;
    long columns_;
    std::vector<SparseEntry> entries_;
    DataDistribution1D columns_distribution_;

public:
    Splitter(long number_of_columns, int number_of_processes):
        columns_distribution_(static_cast<size_t>(number_of_columns), static_cast<size_t>(number_of_processes))
    {}

    void assign_matrix_data(long rows, long columns, std::vector<SparseEntry> &&data) override
    {
        if (rows != columns)
            throw ValueError("Matrix A has non-square size {}x{}", rows, columns);
        rows_ = rows;
        columns_ = columns;
        entries_ = std::move(data);
        std::sort(entries_.begin(), entries_.end(), CSCOrder{});
    }

    std::pair<SparseEntry *, size_t> range_of(int process_rank) override
    {
        // Find the range of triples to store in process proc
        auto first_entry = std::lower_bound(
            entries_.begin(), entries_.end(),
            SparseEntry{-1, static_cast<long>(columns_distribution_.offset(process_rank)), 0},
            CSCOrder{});
        auto last_entry = std::lower_bound(
            entries_.begin(), entries_.end(),
            SparseEntry{-1, static_cast<long>(columns_distribution_.offset(process_rank + 1)), 0},
            CSCOrder{});
        return {&(*first_entry), last_entry - first_entry};
    }

    void free() override
    {
        rows_ = 0;
        columns_ = 0;
        std::vector<SparseEntry>{}.swap(entries_);
    }
};
}

PoweringColAAlgorithm::PoweringColAAlgorithm(const ColASettings &settings):
    settings_(settings)
{}

std::shared_ptr<SparseMatrixSplitter> PoweringColAAlgorithm::init_splitter(long sparse_rows, long sparse_columns)
{
    if (splitter_ == nullptr) {
        assert(sparse_rows == sparse_columns);
        assert(problem_size_ == -1);
        problem_size_ = sparse_rows;
        splitter_ = std::make_shared<Splitter>(sparse_columns, NumberOfProcesses);
    }
    return splitter_;
}

void PoweringColAAlgorithm::initialize(SparseMatrixData &&sparse_part, int dense_seed)
{
    a_ = std::move(sparse_part);
    DataDistribution1D b_distribution(a_.rows, NumberOfProcesses);
    const auto first_column = static_cast<long>(b_distribution.offset(ProcessRank));
    const auto b_part_cols = static_cast<long>(b_distribution.offset(ProcessRank + 1) - first_column);
    const auto b_part_rows = a_.columns;

    spdlog::info("Generating my B part of size {}x{}", b_part_rows, b_part_cols);
    DenseMatrix b_part(b_part_rows, b_part_cols);
    const auto seed = settings_.dense_matrix_seed;
    b_part.in_order_foreach([first_column, seed](auto r, auto c, auto &v) {
        v = generate_double(seed, r, first_column + c);
    });
    std::swap(b_, b_part);

    c_ = DenseMatrix(b_part_rows, b_part_cols);
}


void PoweringColAAlgorithm::replicate()
{
    const int p = NumberOfProcesses;
    // Layer := set of processes having the same part of A
    const int layer_size = settings_.c_param;
    // Ring := set of processes that rotate parts of A during multiplication
    const int ring_size = p / layer_size;

    // Do the math - what is the number of next process in ring  and layer
    const int my_ring_num = ProcessRank / ring_size;
    const int my_layer_num = ProcessRank % ring_size;
    const int first_in_ring = ring_size * my_ring_num;
    world2d_ring_next_ = my_layer_num == ring_size - 1 ? first_in_ring : ProcessRank + 1;
    world2d_ring_prev_ = my_layer_num == 0 ? first_in_ring + ring_size - 1 : ProcessRank - 1;
    spdlog::info("coordinates=({}, {}), prev={}, next={}", my_layer_num, my_ring_num, world2d_ring_prev_,
                 world2d_ring_next_);

    // *** REPLICATE A ***
    // Create temporary communicator for this replication layer
    MPI_Comm layer;
    MPI_Comm_split(MPI_COMM_WORLD, my_layer_num, ProcessRank, &layer);

    // Determine size of combined parts of sparse matrix in this layer and offsets to place them in buffer
    int local_size = static_cast<int>(a_.values.size());
    std::vector<int> all_sizes(layer_size);
    MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, layer);
    int combined_size = 0;
    std::vector<int> all_offsets;
    all_offsets.reserve(layer_size);
    for (const auto s: all_sizes) {
        all_offsets.push_back(combined_size);
        combined_size += s;
    }

    // Serialize the sparse matrix into triples
    std::vector<SparseEntry> local_triples;
    local_triples.reserve(a_.values.size());
    a_.in_order_foreach_nonzero([&local_triples](auto r, auto c, auto v){
        local_triples.push_back({r, c, v});
    });
    assert(local_triples.size() == local_size);

    // Gather all triples within this layer
    MPI_Datatype triple_data_type;
    SparseEntry::InitMPIDataType(triple_data_type);
    std::vector<SparseEntry> combined_triples(combined_size);
    /* this v at the end of MPI_Allgatherv is really an annoying one... */
    MPI_Allgatherv(local_triples.data(), static_cast<int>(local_triples.size()), triple_data_type,
                   combined_triples.data(), all_sizes.data(), all_offsets.data(), triple_data_type,
                   layer);
    spdlog::info("Replication layer {} has {} sparse entries", my_layer_num, combined_size);
    spdlog::trace("Gathered sparse entries: {}", VectorToString(combined_triples));

    // Finally construct the A matrix
    assert(combined_size == combined_triples.size());
    a_ = SparseMatrixData::BuildCSR(a_.rows, a_.columns, combined_triples.size(), combined_triples.data());
    MPI_Type_free(&triple_data_type);
    MPI_Comm_free(&layer);

    // *** PREPARE INBOX FOR ROTATING A ***
    // Create communicator for this ring
    MPI_Comm ring;
    MPI_Comm_split(MPI_COMM_WORLD, my_ring_num, ProcessRank, &ring);

    // Compute maximal size of sparse part within the ring
    int max_combined_size;
    MPI_Allreduce(&combined_size, &max_combined_size, 1, MPI_INT, MPI_MAX, ring);

    MPI_Comm_free(&ring);

    // Initialize inbox structure
    inbox_.rows = a_.rows;
    inbox_.columns = a_.columns;
    inbox_.offsets.resize(a_.offsets.size());
    inbox_.indices.resize(max_combined_size);
    a_.indices.resize(max_combined_size);
    inbox_.values.resize(max_combined_size);
    a_.values.resize(max_combined_size);
}

void PoweringColAAlgorithm::multiply()
{
    assert(a_.columns == inbox_.columns);
    assert(a_.rows == inbox_.rows);
    assert(a_.offsets.size() == inbox_.offsets.size());
    assert(a_.indices.size() == inbox_.indices.size());
    assert(a_.values.size() == inbox_.values.size());

    // Set C = 0
    c_.in_order_foreach([](auto, auto, auto &elt) {
        elt = 0;
    });

    // *** MULTIPLY AND ROTATE A ***
    const int steps = NumberOfProcesses / settings_.c_param;
    spdlog::info("Starting multiplication; it will be executed in {} steps", steps);
    for (int step = 0; step < steps; ++step) {
        // fixme c == 1
        // Asynchronously send sparse part now...
        std::array<MPI_Request, 6> requests;
        MPI_Isend(a_.offsets.data(), static_cast<int>(a_.offsets.size()), MPI_LONG, world2d_ring_next_,
                  Tags::SPARSE_OFFSETS_ARRAY, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(a_.indices.data(), static_cast<int>(a_.offsets.back()), MPI_LONG, world2d_ring_next_,
                  Tags::SPARSE_INDICES_ARRAY, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(a_.values.data(), static_cast<int>(a_.offsets.back()), MPI_DOUBLE, world2d_ring_next_,
                  Tags::SPARSE_VALUES_ARRAY, MPI_COMM_WORLD, &requests[2]);
        // ... and asynchronously receive (note that a_ and inbox_ have the same sizes of all vectors)
        MPI_Irecv(inbox_.offsets.data(), static_cast<int>(a_.offsets.size()), MPI_LONG, world2d_ring_prev_,
                  Tags::SPARSE_OFFSETS_ARRAY, MPI_COMM_WORLD, &requests[3]);
        MPI_Irecv(inbox_.indices.data(), static_cast<int>(inbox_.indices.size()), MPI_LONG, world2d_ring_prev_,
                  Tags::SPARSE_INDICES_ARRAY, MPI_COMM_WORLD, &requests[4]);
        MPI_Irecv(inbox_.values.data(), static_cast<int>(inbox_.values.size()), MPI_DOUBLE, world2d_ring_prev_,
                  Tags::SPARSE_VALUES_ARRAY, MPI_COMM_WORLD, &requests[5]);

        SparseDenseMultiply(a_, b_, c_);

        // Complete exchange of sparse parts
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        spdlog::debug("Finished step {}/{} of multiplication: rotating from {} to {} entries", step + 1, steps,
                      a_.offsets.back(), inbox_.offsets.back());
        std::swap(a_, inbox_);
    }
}

void PoweringColAAlgorithm::swap_cb()
{
    spdlog::debug("Swapping C and B for next multiplication");
    std::swap(b_, c_);
}

std::optional<DenseMatrix> PoweringColAAlgorithm::gather_result()
{
    const auto total_size = problem_size_ * problem_size_;
    std::vector<double> result;
    if (ProcessRank == COORDINATOR_WORLD_RANK)
        result.resize(total_size);

    DataDistribution1D result_distribution(problem_size_, NumberOfProcesses);
    std::vector<int> offsets, sizes;
    if (ProcessRank == COORDINATOR_WORLD_RANK) {
        offsets.reserve(NumberOfProcesses);
        sizes.reserve(NumberOfProcesses);
        for (int i = 0; i < NumberOfProcesses; ++i) {
            auto first_col = result_distribution.offset(i);
            auto last_col = result_distribution.offset(i + 1);
            offsets.push_back(static_cast<int>(first_col * problem_size_));
            sizes.push_back(static_cast<int>((last_col - first_col) * problem_size_));
        }
        spdlog::debug("Gather result offsets are: {}", VectorToString(offsets));
        spdlog::debug("Gather result sizes are: {}", VectorToString(sizes));
    }

    const int my_send_size = static_cast<int>(c_.rows() * c_.columns());
    spdlog::debug("Contributing {} entries to the result", my_send_size);
    MPI_Gatherv(c_.data(), my_send_size, MPI_DOUBLE,
                result.data(), sizes.data(), offsets.data(), MPI_DOUBLE,
                COORDINATOR_WORLD_RANK, MPI_COMM_WORLD);
    spdlog::trace("Sent part of result: {}", VectorToString(c_.data(), c_.data() + my_send_size));

    if (ProcessRank == COORDINATOR_WORLD_RANK)
        return DenseMatrix(problem_size_, problem_size_, std::move(result));
    return std::optional<DenseMatrix>{};
}
