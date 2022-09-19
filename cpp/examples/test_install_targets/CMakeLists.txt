project(SophusInstallTargetTests)

cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 17)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(Sophus 22.99.0 REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

message(WARNING "${Sophus}")

set(example_source hello_so3)

foreach(example_src ${example_source})
  add_executable( ${example_src} ${example_src}.cpp)
  target_link_libraries(${example_src} Eigen3::Eigen fmt::fmt pthread
)
endforeach()

set(example_source2 image_sensor)

foreach(example_src ${example_source2})
  add_executable( ${example_src} ${example_src}.cpp)
  target_link_libraries(${example_src} Eigen3::Eigen fmt::fmt pthread sophus::sophus_image sophus::sophus_sensor
)
endforeach()
