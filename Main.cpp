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


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);
    spdlog::info("Hello from process {}/{}", ProcessRank, NumberOfProcesses);

    try {
        Options = ParseCommandLineOptions(argc - 1, argv + 1);
    } catch (CommandLineError &e) {
        std::cerr << "Invalid command line options: " << e.message() << std::endl;
        return 1;
    }
    SetupLogging();

    SparseMatrix a;
    try {
        std::ifstream s{Options.sparse_matrix_file};
        a = SparseMatrix::Read(s);
    } catch (CSRReadError &e) {
        spdlog::critical("Unable to read CSR file {} with A matrix: {}", Options.sparse_matrix_file.string(),
                         e.message());
        return 1;
    }

    return 0;
}
