cmake_minimum_required(VERSION 3.21)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



# CUDA Options

set(CUDAToolkit_ROOT /home/jupyter-lzj/cuda-12.2)
set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)

# CuTe Path
include_directories(/home/jupyter-lzj/cute_ws/install/include)

project("learn_cute" LANGUAGES CXX CUDA)
add_subdirectory(src)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(demo src/demo.cu)