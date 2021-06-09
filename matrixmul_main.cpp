#include <iostream>
#include <mpi.h>
#include <string_view>
#include <exception>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#include "commons.hpp"

int ProcessRank, NumberOfProcesses;
CommandLineOptions LaunchOptions;


CommandLineOptions ParseCommandLineOptions(const int argc, char **argv)
{
    CommandLineOptions options;
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
            options.used_algorithm = CommandLineOptions::D15_INNER_ABC;
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
        }
    }

    return options;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);

    if (ProcessRank == MASTER_WORLD_RANK)
        spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e][master][%l]  %v", ProcessRank));
    else
        spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e][mpi:{:02}][%l]  %v", ProcessRank));
    spdlog::default_logger()->sinks().clear();
    spdlog::default_logger()->sinks().push_back(std::make_shared<spdlog::sinks::stderr_sink_st>());

    try {
        LaunchOptions = ParseCommandLineOptions(argc, argv);
    } catch (CommandLineError &e) {
        spdlog::error("Invalid command line options: {}", e.message());
    }

    spdlog::info("Hello from process {}/{}", ProcessRank, NumberOfProcesses);
    return 0;
}
