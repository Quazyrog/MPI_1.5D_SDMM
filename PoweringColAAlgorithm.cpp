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

void PoweringColAAlgorithm::initialize(SparseMatrixData &&sparse_part)
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
    const int layer_size = settings_.c_param;  /* processes in the same layer replicate data */
    const int ring_size = p / layer_size; /* processes in one ring rotate data */

    // *** CREATE CARTESIAN COMM ***
    // Arrange processes in a cylinder of p/layer_size layer_size
    int coords[] = {layer_size, ring_size};
    int periodic[] =  {false, true};
    spdlog::debug("Creating cylindrical communicator of {} layer_size and {} layer_size", ring_size, layer_size);
    MPI_Cart_create(MPI_COMM_WORLD, 2, coords, periodic, true, &world2d_);

    // Query the new communicator and save some info
    MPI_Comm_rank(world2d_, &world2d_my_rank_);
    MPI_Cart_coords(world2d_, world2d_my_rank_, 2, world2d_my_coords_);
    coords[0] = world2d_my_coords_[0];
    coords[1] = world2d_my_coords_[1] + 1; /* next */
    MPI_Cart_rank(world2d_, coords, &world2d_ring_next_);
    coords[1] = world2d_my_coords_[1] - 1; /* prev */
    MPI_Cart_rank(world2d_, coords, &world2d_ring_prev_);
    coords[1] = 0; /* local coordinator, just in case */
    MPI_Cart_rank(world2d_, coords, &world2d_ring_coordinator_);
    spdlog::info("Cartesian communicator initialized: coords=({}, {}), prev={}, next={}, ring_coordinator={}",
                 world2d_my_coords_[0], world2d_my_coords_[1], world2d_ring_prev_, world2d_ring_next_,
                 world2d_ring_coordinator_);

    // *** REPLICATE A ***
    // Create temporary communicator for this replication layer
    MPI_Comm layer;
    int remain_dims[] = {true, false};
    MPI_Cart_sub(world2d_, remain_dims, &layer);

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
    spdlog::info("Replication layer {} has {} sparse entries", world2d_my_coords_[1], combined_size);

    // Finally construct the A matrix
    assert(combined_size == combined_triples.size());
    a_ = SparseMatrixData::BuildCSR(a_.rows, a_.columns, combined_triples.size(), combined_triples.data());
    MPI_Type_free(&triple_data_type);
    MPI_Comm_free(&layer);

    // *** PREPARE INBOX FOR ROTATING A ***
    // Create communicator for this ring
    MPI_Comm ring;
    remain_dims[0] = false;
    remain_dims[1] = true;
    MPI_Cart_sub(world2d_, remain_dims, &ring);

    // Compute maximal size of sparse part within the ring
    int max_combined_size;
    MPI_Allreduce(&combined_size, &max_combined_size, 1, MPI_INT, MPI_MAX, ring);

    // Initialize inbox structure
    inbox_.rows = a_.rows;
    inbox_.columns = a_.columns;
    inbox_.offsets.resize(a_.offsets.size());
    inbox_.indices.resize(max_combined_size);
    a_.indices.resize(max_combined_size);
    inbox_.values.resize(max_combined_size);
    a_.values.resize(max_combined_size);
}

PoweringColAAlgorithm::~PoweringColAAlgorithm()
{
    MPI_Comm_free(&world2d_);
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
    if (ProcessRank == COORDINATOR_WORLD_RANK)
        spdlog::info("Starting multiplication; it will be executed in {} steps", steps);
    for (int step = 0; step < steps; ++step) {
        // fixme c == 1
        // Asynchronously send sparse part now...
        std::array<MPI_Request, 6> requests;
        MPI_Isend(a_.offsets.data(), static_cast<int>(a_.offsets.size()), MPI_LONG, world2d_ring_next_,
                  Tags::SPARSE_OFFSETS_ARRAY, world2d_, &requests[0]);
        MPI_Isend(a_.indices.data(), static_cast<int>(a_.offsets.back()), MPI_LONG, world2d_ring_next_,
                  Tags::SPARSE_INDICES_ARRAY, world2d_, &requests[1]);
        MPI_Isend(a_.values.data(), static_cast<int>(a_.offsets.back()), MPI_DOUBLE, world2d_ring_next_,
                  Tags::SPARSE_VALUES_ARRAY, world2d_, &requests[2]);
        // ... and asynchronously receive (note that a_ and inbox_ have the same sizes of all vectors)
        MPI_Irecv(inbox_.offsets.data(), static_cast<int>(a_.offsets.size()), MPI_LONG, world2d_ring_prev_,
                  Tags::SPARSE_OFFSETS_ARRAY, world2d_, &requests[3]);
        MPI_Irecv(inbox_.indices.data(), static_cast<int>(inbox_.indices.size()), MPI_LONG, world2d_ring_prev_,
                  Tags::SPARSE_INDICES_ARRAY, world2d_, &requests[4]);
        MPI_Irecv(inbox_.values.data(), static_cast<int>(inbox_.values.size()), MPI_DOUBLE, world2d_ring_prev_,
                  Tags::SPARSE_VALUES_ARRAY, world2d_, &requests[5]);

        SparseDenseMultiply(a_, b_, c_);

        // Complete exchange of sparse parts
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        spdlog::debug("Finished step {}/{} of multiplication: rotating from {} to {} entries", step + 1, steps,
                      a_.offsets.back(), inbox_.offsets.back());
        std::swap(a_, inbox_);
    }
}
