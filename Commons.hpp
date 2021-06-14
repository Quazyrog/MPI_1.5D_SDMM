#ifndef COMMONS_HPP
#define COMMONS_HPP

#include <optional>
#include <algorithm>
#include <mpi.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

static constexpr int COORDINATOR_WORLD_RANK = 0;
extern int ProcessRank, NumberOfProcesses;

struct ProgramOptions
{
    enum Algorithm { D15_COL_A, D15_INNER_ABC };

    std::string sparse_matrix_file;
    int dense_matrix_seed;
    int replication_group_size;
    int exponent;
    Algorithm used_algorithm = D15_COL_A;

    bool print_result = false;
    bool print_ge_count;
    double count_compare_number = false;

    spdlog::level::level_enum stderr_log_level = spdlog::level::level_enum::info;
    std::string log_file_path;
    bool verify = false;
};


class DataDistribution1D
{
    size_t base_chunk_size_;
    size_t remainder_;

public:
    DataDistribution1D(size_t size, size_t parts_number)
    {
        base_chunk_size_ = size / parts_number;
        remainder_ = size - parts_number * base_chunk_size_;
    }

    inline size_t offset(size_t part) const noexcept
    {
        return part * base_chunk_size_ + std::min(part, remainder_);
    }

    inline size_t part(size_t index) const noexcept
    {
        auto large_parts_before_index = index / (base_chunk_size_ + 1);
        if (large_parts_before_index < remainder_)
            return large_parts_before_index;
        index -= remainder_ * (base_chunk_size_ + 1);
        return remainder_ + index / base_chunk_size_;
    }
};


class Error: public std::exception
{
protected:
    std::string _message;
    std::string _what = "Error";

    template<class ...Args>
    Error(const char *class_name, const char *format, Args... args):
        _message(fmt::format(format, std::forward<Args>(args)...))
    {
            _what = class_name + (": " + _message);
    }

public:
    const std::string &message() noexcept { return _message; }
    const char *what() const noexcept override { return _message.c_str(); }
};

class CommandLineError: public Error
{
public:
    template<class ...Args>
    explicit CommandLineError(const char *format, Args... args):
        Error("CommandLineError", format, std::forward<Args>(args)...)
    {}
};


class IOError: public Error
{
protected:
    std::string file_name_;

    using Error::Error;

public:
    template<class ...Args>
    explicit IOError(const char *format, Args... args):
        Error("IOError", format, std::forward<Args>(args)...)
    {}

    const auto &file_name() const
    { return file_name_; }
    void set_file_name(std::string fn)
    { file_name_ = std::move(fn); }
};

class ValueError: public Error
{
public:
    template<class ...Args>
    explicit ValueError(const char *format, Args... args):
        Error("ValueError", format, std::forward<Args>(args)...)
    {}
};

class NotImplementedError: public Error
{
public:
    template<class ...Args>
    explicit NotImplementedError(const char *format, Args... args):
        Error("NotImplementedError", format, std::forward<Args>(args)...)
    {}
};

class MPIError: public Error
{
    int code_;
    std::string function_name_;

public:
    static std::string ErrorString(int code)
    {
        char str[MPI_MAX_ERROR_STRING + 1];
        int len = 0;
        auto res = MPI_Error_string(code, str, &len);
        if (res != MPI_SUCCESS)
            return "(invalid error code)";
        str[len] = 0;
        return str;
    }

    template<class ...Args>
    explicit MPIError(const char *format, Args... args):
        Error("MPIError", format, std::forward<Args>(args)...)
    {}

    explicit MPIError(int error_code, const char *call):
        Error("MPIError", "MPI call {} returned error #{}: {}", call, error_code, ErrorString(error_code)),
        function_name_(call),
        code_(error_code)
    {}

    std::string error_string() const
    {
        return ErrorString(code_);
    }

    int error_code() const noexcept
    {
        return code_;
    }

    const std::string &function_name() const noexcept
    {
        return function_name_;
    }
};


namespace Tags {
enum MessageTag {
    SPARSE_OFFSETS_ARRAY,
    SPARSE_INDICES_ARRAY,
    SPARSE_VALUES_ARRAY,
};
}


#endif //COMMONS_HPP
