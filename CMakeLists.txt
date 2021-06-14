cmake_minimum_required(VERSION 3.10)
project(zadanie_)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
add_subdirectory(external/fmt)
find_package(fmt REQUIRED)
add_subdirectory(external/spdlog)
find_package(spdlog REQUIRED)


add_executable(generator generator_main.cpp densematgen.cpp densematgen.h)
target_compile_options(generator PUBLIC -std=c++14  -g3 -Wall -O0)

add_executable(matrixmul Main.cpp
        Commons.hpp
        Debug.hpp
        Matrix.cpp Matrix.hpp
        densematgen.cpp densematgen.h
        PoweringAlgorithm.hpp PoweringAlgorithm.cpp
        PoweringAlgorithmColA.cpp PoweringAlgorithmColA.hpp
        PoweringAlgorithmInnerAbc.cpp PoweringAlgorithmInnerAbc.hpp
        )
target_compile_definitions(matrixmul PRIVATE -DSPDLOG_COMPILED_LIB)
target_include_directories(matrixmul PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(matrixmul BEFORE PRIVATE external/fmt/include external/spdlog/include)
target_link_libraries(matrixmul ${MPI_LIBRARIES} fmt::fmt spdlog::spdlog)
if(MPI_COMPILE_FLAGS)
    set_target_properties(matrixmul PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(matrixmul PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()