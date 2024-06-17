cmake_minimum_required(VERSION 3.24)

project(CUDAVectorAdd)

enable_language(CUDA) # Enable CUDA language support

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES
    52
    60
    61
    70
    75
    80
    86)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_executable(kernel main.cpp kernel.cu)

set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kernel PRIVATE cuda)
