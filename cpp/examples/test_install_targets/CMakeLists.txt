project(SophusInstallTargetTests)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(Sophus 22.99.0 REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

set(example_source hello_so3)

foreach(example_src ${example_source})
  add_executable( ${example_src} ${example_src}.cpp)
  target_link_libraries(${example_src}
   Eigen3::Eigen
   fmt::fmt
   pthread
   Sophus::sophus
)
endforeach()

set(example_source2 image_sensor)

foreach(example_src ${example_source2})
  add_executable( ${example_src} ${example_src}.cpp)
  target_link_libraries(${example_src} Eigen3::Eigen fmt::fmt pthread Sophus::sophus_image Sophus::sophus_sensor
)
endforeach()

if (BUILD_FARM_NG_PROTOS)
  find_package(Protobuf REQUIRED)
  add_executable(compile_all_proto compile_all_proto.cpp)
  target_link_libraries(compile_all_proto PUBLIC
    fmt::fmt
    Eigen3::Eigen
    protobuf::libprotobuf
    farm_ng_core::farm_ng_core_prototools
    farm_ng_core::farm_ng_core_misc_proto_conv
    Sophus::sophus_sensor_proto_conv
    Sophus::sophus_image_proto_conv
    )
endif()
