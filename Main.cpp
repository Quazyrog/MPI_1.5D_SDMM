#include <iostream>
#include <mpi.h>
#include <string_view>
#include <exception>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <fstream>
#include <spdlog/sinks/ansicolor_sink.h>
#include <unistd.h>
#include "Commons.hpp"
#include "Matrix.hpp"
#include "MatrixPoweringAlgorithm.hpp"
#include "PoweringColAAlgorithm.hpp"
#include "densematgen.h"
#include "PoweringInnerAbcAlgorithm.hpp"
#include "Debug.hpp"

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
        } else if (option == "--verify") {
            options.verify = true;
        } else {
            throw CommandLineError("unknown option '{}'", option);
        }
    }

    if (auto level = std::getenv("LOG_VERBOSITY")) {
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
    if (auto pattern = std::getenv("LOG_FILE_PATTERN")) {
        std::string log_file = pattern;
        auto pos = log_file.find('%');
        if (pos == std::string::npos) {
            std::cerr << "WARNING: invalid log file name pattern (no % char found)";
        } else {
            log_file.replace(pos, 1, std::to_string(ProcessRank));
            options.log_file_path = log_file;
        }
    }

    return options;
}


void SetupLogging()
{
    spdlog::set_level(spdlog::level::level_enum::trace);
    std::string pattern;
    if (ProcessRank == COORDINATOR_WORLD_RANK)
        pattern = fmt::format("[%Y-%m-%d %H:%M:%S.%e] [master] [%l] :: %v", ProcessRank);
    else
        pattern = fmt::format("[%Y-%m-%d %H:%M:%S.%e] [mpi:{:02}] [%l] :: %v", ProcessRank);

    spdlog::sink_ptr default_sink;
    if (Options.log_file_path.empty())
        default_sink = std::make_shared<spdlog::sinks::stderr_sink_st>();
    else
        default_sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(Options.log_file_path, true);
    default_sink->set_level(Options.stderr_log_level);
    default_sink->set_pattern(pattern);

    spdlog::default_logger()->sinks().clear();
    spdlog::default_logger()->sinks().push_back(default_sink);
    if (ProcessRank == COORDINATOR_WORLD_RANK && !Options.log_file_path.empty()) {
        auto console_logger = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>();
        console_logger->set_pattern(pattern);
        console_logger->set_level(Options.stderr_log_level);
        spdlog::default_logger()->sinks().push_back(console_logger);
    }
}


auto InitializeAlgorithm(MatrixPoweringAlgorithm &algorithm)
{
    long a_mat_size[2];
    SparseMatrixData a_part;

    // Initialize A
    if (ProcessRank == COORDINATOR_WORLD_RANK) {
        spdlog::info("Starting initial matrix propagation");

        // Prepare A data as triples
        std::ifstream s{Options.sparse_matrix_file};
        auto [mat_rows, mat_cols, mat_data] = SparseMatrixData::ReadCSRFile(s);
        spdlog::info("Read {} entries from CSR file '{}'", mat_data.size(), Options.sparse_matrix_file.string());
        auto splitter = algorithm.init_splitter(static_cast<long>(mat_rows), static_cast<long>(mat_cols));
        splitter->assign_matrix_data(static_cast<long>(mat_rows), static_cast<long>(mat_cols), std::move(mat_data));

        // Broadcast A dimension
        spdlog::debug("Broadcasting matrix A size: {}x{}", mat_rows, mat_cols);
        a_mat_size[0] = static_cast<long>(mat_rows);
        a_mat_size[1] = static_cast<long>(mat_cols);
        MPI_Bcast(a_mat_size, 2, MPI_LONG, COORDINATOR_WORLD_RANK, MPI_COMM_WORLD);

        // Distribute the sparse matrix A
        for (int proc = 0; proc < NumberOfProcesses; ++proc) {
            // Construct and send the matrix in CSR representation
            auto [entries, count] = splitter->range_of(proc);
            spdlog::trace("Initial part of process {} is: {}", proc, Debug::VectorToString(entries, entries + count));
            auto mat_part = SparseMatrixData::BuildCSR(static_cast<long>(mat_rows), static_cast<long>(mat_cols),
                                                       count, entries);
            if (proc != ProcessRank) {
                MPI_Send(mat_part.offsets.data(), static_cast<int>(mat_part.offsets.size()), MPI_LONG, proc,
                         Tags::SPARSE_OFFSETS_ARRAY, MPI_COMM_WORLD);
                MPI_Send(mat_part.indices.data(), static_cast<int>(mat_part.indices.size()), MPI_LONG, proc,
                         Tags::SPARSE_INDICES_ARRAY, MPI_COMM_WORLD);
                MPI_Send(mat_part.values.data(), static_cast<int>(mat_part.values.size()), MPI_DOUBLE, proc,
                         Tags::SPARSE_VALUES_ARRAY, MPI_COMM_WORLD);
                spdlog::debug("Sent {} elements to {} process", mat_part.values.size(), proc);
            } else {
                spdlog::debug("Storing {} elements in my A matrix part", mat_part.values.size());
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
        spdlog::trace("Received offsets array: {}", Debug::VectorToString(a_part.offsets));
        spdlog::debug("Number of nonzero elements in my part of A is {}", a_part.offsets.back());

        a_part.indices.resize(a_part.offsets.back());
        MPI_Recv(a_part.indices.data(), static_cast<int>(a_part.indices.size()), MPI_LONG, COORDINATOR_WORLD_RANK,
                 Tags::SPARSE_INDICES_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        spdlog::trace("Received indices array: {}", Debug::VectorToString(a_part.indices));

        a_part.values.resize(a_part.offsets.back());
        MPI_Recv(a_part.values.data(), static_cast<int>(a_part.values.size()), MPI_DOUBLE, COORDINATOR_WORLD_RANK,
                 Tags::SPARSE_VALUES_ARRAY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        spdlog::trace("Received values array: {}", Debug::VectorToString(a_part.values));
    }
    /* now every process holds it's range of columns of A without replication */
    algorithm.initialize(std::move(a_part), Options.dense_matrix_seed);
    spdlog::info("Algorithm initialization complete!");
}

static double RoundWallTime(double seconds)
{
    return std::round(seconds * 1'000'000) / 1'000;
}

ColumnMajorMatrix TestCompute()
{
    spdlog::info("(TEST) Reading sparse matrix from {}", Options.sparse_matrix_file.string());
    std::ifstream s{Options.sparse_matrix_file};
    auto [r, c, entries] = SparseMatrixData::ReadCSRFile(s);
    if (r != c) {
        spdlog::critical("(TEST) Sparse matrix was supposed to be square, not {}x{}!", r, c);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    auto n = r;
    auto sparse = SparseMatrixData::BuildCSR(n, n, entries.size(), entries.data());
    ColumnMajorMatrix dense(n, n), result(n, n);
    dense.in_order_foreach([](auto r, auto c, auto &v) {
        v = generate_double(Options.dense_matrix_seed, r, c);
    });

    for (int pow = 0; pow < Options.exponent; ++pow) {
        if (pow != 0) {
            std::swap(dense, result);
            result.in_order_foreach([](auto, auto, auto &v) { v = 0; });
        }
        spdlog::info("(TEST) Executing multiplication {}/{}", pow + 1, Options.exponent);
        SparseDenseMultiply(sparse, dense, result);
    }

    return result;
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
    spdlog::info("Built: {} {}", __DATE__, __TIME__);
    if constexpr (Debug::ENABLED)
        spdlog::warn("THIS BUILD HAS ENABLED DEBUG FUNCTIONALITY THAT IS BAD FOR PERFORMANCE!");
    spdlog::info("My PID is {}", getpid());
    spdlog::info("Running on {} tasks; coordinator rank is {}", NumberOfProcesses, COORDINATOR_WORLD_RANK);

    double initialization_duration;
    MatrixPoweringAlgorithm *algorithm;
    try {
        if (Options.used_algorithm == ProgramOptions::D15_COL_A) {
            ColASettings settings;
            settings.dense_matrix_seed = Options.dense_matrix_seed;
            settings.c_param = Options.replication_group_size;
            algorithm = new PoweringColAAlgorithm(settings);
        } else {
            algorithm = new PoweringInnerABCAlgorithm(Options.replication_group_size);
        }

        // Initialize
        initialization_duration = MPI_Wtime();
        InitializeAlgorithm(*algorithm);
        MPI_Barrier(MPI_COMM_WORLD);
        initialization_duration = MPI_Wtime() - initialization_duration;
        spdlog::info("Initialization completed in {}ms", RoundWallTime(initialization_duration));
    } catch (CSRReadError &e) {
        spdlog::critical("Unable to read CSR file {} with A matrix: {}", Options.sparse_matrix_file.string(),
                         e.message());
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (std::exception &e) {
        spdlog::critical("Failed to initialize the algorithm: {}", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Replicate A matrix
    double replication_duration = MPI_Wtime();
    algorithm->replicate();
    MPI_Barrier(MPI_COMM_WORLD);
    replication_duration = MPI_Wtime() - replication_duration;
    spdlog::info("Replication completed in {}ms", RoundWallTime(replication_duration));

    // Do the powering
    double multiplication_duration = MPI_Wtime();
    if (Options.exponent == 0) {
        spdlog::critical("Sorry, I'm not ready for exponent=0!");
        MPI_Finalize();
        return 2;
    }
    for (int pow = 0; pow < Options.exponent; ++pow) {
        if (pow != 0)
            algorithm->swap_cb();
        spdlog::info("Executing multiplication {}/{}", pow + 1, Options.exponent);
        algorithm->multiply();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    multiplication_duration = MPI_Wtime() - multiplication_duration;
    spdlog::info("Multiplication completed in {}ms", RoundWallTime(multiplication_duration));

    ColumnMajorMatrix result;
    if (Options.print_result || Options.verify) {
        spdlog::info("Gathering the result matrix");
        if (auto res = algorithm->gather_result()) {
            spdlog::info("I have entire result matrix");
            result = *res;
        }
    }

    if (ProcessRank == COORDINATOR_WORLD_RANK) {
        if (Options.verify) {
            constexpr double EPS = 1E-9;
            auto expected = TestCompute();
            bool fail = false;

            // Expected
            for (long r = 0; r < expected.rows(); ++r) {
                for (long c = 0; c < expected.columns(); ++c) {
                    if (std::abs(result(r, c) - expected(r, c)) > EPS) {
                        if (Options.print_result) {
                            spdlog::warn("At position ({}, {}) value {} was expected, but {} found", r, c,
                                         expected(r, c), result(r, c));
                        }
                        fail = true;
                    }
                }
            }

            if (fail)
                spdlog::error("The result seems wrong!");
            else
                spdlog::info("The result seems right...");

        } else {
            if (Options.print_result) {
                std::cout << result.rows() << " " << result.columns();
                for (long r = 0; r < result.rows(); ++r) {
                    std::cout << "\n";
                    for (long c = 0; c < result.columns(); ++c)
                        std::cout << result(r, c) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    spdlog::info("So Long, and Thanks for All the Fish!");
    delete algorithm;
    MPI_Finalize();
    return 0;
}
