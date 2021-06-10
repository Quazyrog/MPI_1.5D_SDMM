#include <iostream>
#include <mpi.h>
#include <string_view>
#include <exception>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <fstream>
#include "Commons.hpp"
#include "SparseMatrix.hpp"

int ProcessRank, NumberOfProcesses;
ProgramOptions Options;


ProgramOptions ParseCommandLineOptions(const int argc, char **argv)
{
    ProgramOptions options;
    int arg_index = 0;
    auto consume_arg = [&arg_index, argc, argv](const char *option = nullptr) {
        if (arg_index >= argc) {
            if (option)
                throw CommandLineError("option {} requires a value", option);
            throw CommandLineError("more arguments expected");
        }
        return std::string(argv[++arg_index]);
    };
    auto convert_int = [](const std::string &s, const char *option) {
        try {
            size_t pos = 0;
            auto val = std::stoi(s, &pos);
            if (pos != s.length())
                throw std::invalid_argument("invalid numeral");
            return val;
        } catch (std::exception &e) {
            throw CommandLineError("invalid int literal '{}' for option {}: {}", s, option, e.what());
        }
    };

    for (; arg_index < argc; ++arg_index) {
        auto option = std::string(argv[arg_index]);
        if (option == "-f") {
            options.sparse_matrix_file = consume_arg("-f");
        } else if (option == "-s") {
            options.dense_matrix_seed = convert_int(consume_arg("-s"), "-s");
        } else if (option == "-c") {
            options.replication_group_size = convert_int(consume_arg("-c"), "-c");
        } else if (option == "-e") {
            options.exponent = convert_int(consume_arg("-e"), "-e");
        } else if (option == "-i") {
            options.used_algorithm = ProgramOptions::D15_INNER_ABC;
        } else if (option == "-v") {
            options.print_result = true;
        } else if (option == "-g") {
            auto s = consume_arg("-g");
            try {
                size_t pos = 0;
                auto val = std::stod(s, &pos);
                if (pos != s.length())
                    throw std::invalid_argument("invalid numeral");
                options.print_ge_count = val;
            } catch (std::exception &e) {
                throw CommandLineError("invalid double literal '{}' for option -g: {}", s, e.what());
            }
        } else {
            throw CommandLineError("unknown option '{}'", option);
        }
    }

    if (auto level = std::getenv("STDERR_VERBOSITY")) {
#define MATRIXMUL_MAIN_CONVERT_LEVEL(level_name) \
    if (strcmp(level, #level_name) == 0) options.stderr_log_level = spdlog::level::level_enum:: level_name
        MATRIXMUL_MAIN_CONVERT_LEVEL(trace);
        MATRIXMUL_MAIN_CONVERT_LEVEL(debug);
        MATRIXMUL_MAIN_CONVERT_LEVEL(info);
        MATRIXMUL_MAIN_CONVERT_LEVEL(warn);
        MATRIXMUL_MAIN_CONVERT_LEVEL(err);
        MATRIXMUL_MAIN_CONVERT_LEVEL(critical);
        MATRIXMUL_MAIN_CONVERT_LEVEL(off);
    }

    return options;
}


void SetupLogging()
{
    spdlog::default_logger()->sinks().clear();
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_sink_st>();
    if (ProcessRank == COORDINATOR_WORLD_RANK)
        stderr_sink->set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [master] [%l] :: %v", ProcessRank));
    else
        stderr_sink->set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [mpi:{:02}] [%l] :: %v", ProcessRank));
    spdlog::set_level(spdlog::level::level_enum::trace);
    stderr_sink->set_level(Options.stderr_log_level);
    spdlog::default_logger()->sinks().push_back(stderr_sink);
}


/**
 * Prepare initial distribution of matrices A and B among processes.
 * @return
 */
auto InitializeMatrices()
{
    long a_mat_size[2];
    SparseMatrixData a_part;
    spdlog::info("Starting initial matrix propagation");

    // Initialize A
    if (ProcessRank == COORDINATOR_WORLD_RANK) {
        // Prepare A data as triples
        std::ifstream s{Options.sparse_matrix_file};
        auto [mat_rows, mat_cols, mat_data] = SparseMatrixData::ReadCSRFile(s);
        std::sort(mat_data.begin(), mat_data.end(), CSCOrder{});
        spdlog::info("Read {} entries from CSR file '{}'", mat_data.size(), Options.sparse_matrix_file.string());

        // Broadcast A dimension
        spdlog::debug("Broadcasting matrix A size: {}x{}", mat_rows, mat_cols);
        a_mat_size[0] = static_cast<long>(mat_rows);
        a_mat_size[1] = static_cast<long>(mat_cols);
        MPI_Bcast(a_mat_size, 2, MPI_LONG, COORDINATOR_WORLD_RANK, MPI_COMM_WORLD);

        // Distribute A columns-wise
        DataDistribution1D a_distribution(mat_cols, NumberOfProcesses);
        for (int proc = 0; proc < NumberOfProcesses; ++proc) {
            // Find the range of triples to store in process proc
            auto first_entry = std::lower_bound(
                mat_data.begin(), mat_data.end(),
                SparseEntry{-1, static_cast<long>(a_distribution.offset(proc)), 0},
                CSCOrder{});
            auto last_entry = std::lower_bound(
                mat_data.begin(), mat_data.end(),
                SparseEntry{-1, static_cast<long>(a_distribution.offset(proc + 1)), 0},
                CSCOrder{});

            // Construct and send the matrix in CSR representation
            auto mat_part = SparseMatrixData::BuildCSR(static_cast<long>(mat_rows), static_cast<long>(mat_cols),
                                                       last_entry - first_entry, &(*first_entry));
            if (proc != ProcessRank) {
                MPI_Send(mat_part.offsets.data(), static_cast<int>(mat_part.offsets.size()), MPI_LONG, proc,
                         Tags::SPARSE_OFFSETS_ARRAY, MPI_COMM_WORLD);
                MPI_Send(mat_part.indices.data(), static_cast<int>(mat_part.indices.size()), MPI_LONG, proc,
                         Tags::SPARSE_INDICES_ARRAY, MPI_COMM_WORLD);
                MPI_Send(mat_part.values.data(), static_cast<int>(mat_part.values.size()), MPI_DOUBLE, proc,
                         Tags::SPARSE_VALUES_ARRAY, MPI_COMM_WORLD);
                spdlog::debug("Sent {} elements to {} process", mat_part.values.data(), proc);
            } else {
                spdlog::debug("Storing {} elements in my A matrix part", mat_part.values.data());
                a_part = std::move(mat_part);
            }
        }

        // Done
        spdlog::info("Initial matrix propagation finished for A");

    } else {
        // Receive A matrix size and initialize CSR metadata
        MPI_Bcast(a_mat_size, 2, MPI_LONG, COORDINATOR_WORLD_RANK, MPI_COMM_WORLD);
        spdlog::debug("Received A size {}x{}", a_mat_size[0], a_mat_size[1]);
        a_part.rows = a_mat_size[0];
        a_part.columns = a_mat_size[1];
        a_part.offsets.resize(a_part.rows + 1);

        // Allocate and receive my A part in CSR form
        MPI_Recv(a_part.offsets.data(), static_cast<int>(a_part.offsets.size()), MPI_LONG, COORDINATOR_WORLD_RANK,
                 Tags::SPARSE_OFFSETS_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        spdlog::trace("Received offsets array: {}", VectorToString(a_part.offsets));
        spdlog::debug("Number of nonzero elements in my part of A is {}", a_part.offsets.back());

        a_part.indices.resize(a_part.offsets.back());
        MPI_Recv(a_part.indices.data(), static_cast<int>(a_part.indices.size()), MPI_LONG, COORDINATOR_WORLD_RANK,
                 Tags::SPARSE_INDICES_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        spdlog::trace("Received indices array: {}", VectorToString(a_part.indices));

        a_part.values.resize(a_part.offsets.back());
        MPI_Recv(a_part.values.data(), static_cast<int>(a_part.values.size()), MPI_DOUBLE, COORDINATOR_WORLD_RANK,
                 Tags::SPARSE_VALUES_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        spdlog::trace("Received values array: {}", VectorToString(a_part.values));
    }
    // Now every process holds it's range of columns of A without replication

    return a_part;
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);

    try {
        Options = ParseCommandLineOptions(argc - 1, argv + 1);
    } catch (CommandLineError &e) {
        std::cerr << "Invalid command line options: " << e.message() << std::endl;
        return 1;
    }
    SetupLogging();


    try {
        InitializeMatrices();
    } catch (CSRReadError &e) {
        spdlog::critical("Unable to read CSR file {} with A matrix: {}", Options.sparse_matrix_file.string(),
                         e.message());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
