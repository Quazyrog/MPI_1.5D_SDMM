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
    generate_dense_part_();
}

void PoweringColAAlgorithm::generate_dense_part_()
{
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
}
