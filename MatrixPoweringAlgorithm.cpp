#include "MatrixPoweringAlgorithm.hpp"
#include "Debug.hpp"

void MatrixPoweringAlgorithm::init_inbox_(MPI_Comm &ring)
{
    int combined_size = static_cast<int>(a_.size());
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

void MatrixPoweringAlgorithm::replicate_a_(MPI_Comm &layer, const int layer_size, const int layer_num)
{
    // Gather sizes
    std::vector<int> sizes(layer_size);
    int size = static_cast<int>(a_.size());
    MPI_Allgather(&size, 1, MPI_INT, sizes.data(), 1, MPI_INT, layer);

    // Prepare offsets for Allgatherv and compute combined size
    std::vector<int> offsets;
    offsets.reserve(layer_size);
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
    spdlog::trace("Layer {} offsets array: {}", layer_num, Debug::VectorToString(offsets));
    MPI_Datatype sparse_entry_datatype;
    SparseEntry::InitMPIDataType(sparse_entry_datatype);
    std::vector<SparseEntry> gathered_entries(total_size);
    MPI_Allgatherv(local_entries.data(), static_cast<int>(local_entries.size()), sparse_entry_datatype,
                   gathered_entries.data(), sizes.data(), offsets.data(), sparse_entry_datatype, layer);

    // Reconstruct new matrix
    spdlog::info("Layer {} has {} entries in its sparse matrix", layer_num, total_size);
    spdlog::trace("Gathered entries: {}", Debug::VectorToString(gathered_entries));
    a_ = SparseMatrixData::BuildCSR(a_.rows, a_.columns, gathered_entries.size(), gathered_entries.data());
}
