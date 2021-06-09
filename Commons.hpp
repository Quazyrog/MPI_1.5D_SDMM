#ifndef COMMONS_HPP
#define COMMONS_HPP

#include <filesystem>
#include <optional>
#include <fmt/format.h>

static constexpr int COORDINATOR_WORLD_RANK = 0;

struct ProgramOptions
{
    enum Algorithm { D15_COL_A, D15_INNER_ABC };

    std::filesystem::path sparse_matrix_file;
    int dense_matrix_seed;
    int replication_group_size;
    int exponent;
    Algorithm used_algorithm = D15_COL_A;

    bool print_result = false;
    std::optional<double> print_ge_count;

    spdlog::level::level_enum stderr_log_level;
};


class Error: public std::exception
{
protected:
    std::string _message;
    std::string _what = "Error";

    template<class ...Args>
    Error(const char *class_name, std::string_view format, Args... args):
        _message(fmt::format(format, std::forward<Args>(args)...))
    {
            _what = class_name + _message;
    }

public:
    const std::string &message() noexcept { return _message; }
    const char *what() const noexcept override { return _message.c_str(); }
};

class CommandLineError: public Error
{
public:
    template<class ...Args>
    explicit CommandLineError(std::string_view format, Args... args):
        Error("CommandLineError", format, std::forward<Args>(args)...)
    {}
};


class IOError: public Error
{
protected:
    std::filesystem::path file_name_;

    using Error::Error;

public:
    template<class ...Args>
    explicit IOError(std::string_view format, Args... args):
        Error("IOError", format, std::forward<Args>(args)...)
    {}

    const auto &file_name() const
    { return file_name_; }
    void set_file_name(std::filesystem::path fn)
    { file_name_ = std::move(fn); }
};



#endif //COMMONS_HPP
