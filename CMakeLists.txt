cmake_minimum_required(VERSION 3.16)
project(Sophus VERSION 22.99.0)


find_package(farm_ng_cmake 0.1.0 REQUIRED)
farm_ng_module()

find_package(Eigen3 3.4.0 REQUIRED)
find_package(farm_ng_core 0.1.0 REQUIRED)

farm_ng_enable_testing()
add_subdirectory(protos)
add_subdirectory(cpp)

farm_ng_export_module(
  NAME Sophus
  REQUIRED_DEPS farm_ng_core fmt)
