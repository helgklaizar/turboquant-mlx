cmake_minimum_required(VERSION 3.24)
project(polarquant_ext)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)
find_package(mlx REQUIRED)

pybind11_add_module(polarquant_ext polarquant_ext.cpp)
target_link_libraries(polarquant_ext PRIVATE mlx)

# You can compile the metal file using the exact steps 
# required by your version of the MLX framework.
