#ifndef COMMONS_HPP
#define COMMONS_HPP

#include <filesystem>
#include <optional>
#include <algorithm>
#include <fmt/format.h>

static constexpr int COORDINATOR_WORLD_RANK = 0;
extern int ProcessRank, NumberOfProcesses;

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

class ValueError: public Error
{
public:
    template<class ...Args>
    explicit ValueError(std::string_view format, Args... args):
        Error("ValueError", format, std::forward<Args>(args)...)
    {}
};


namespace Tags {
enum MessageTag {
    SPARSE_OFFSETS_ARRAY,
    SPARSE_INDICES_ARRAY,
    SPARSE_VALUES_ARRAY,
};
}

template<class T>
std::string VectorToString(const std::vector<T> &vec)
{
    std::stringstream s;
    s << "[";
    bool comma = false;
    for (const auto &v: vec) {
        if (comma)
            s << ", ";
        s << v;
        comma = true;
    }
    s << "]";
    return s.str();
}

#endif //COMMONS_HPP
