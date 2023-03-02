project(SophusInstallTargetTests)

cmake_minimum_required(VERSION 3.16)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Coarsely, the c++ standard used is c++17 plus concepts from c++20.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  # For clang, c++20 is enabled, but only concepts shall be used from the features newly added.
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wsign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wreorder-ctor -Wreorder-init-list")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-variable -Wno-unused-function -Wno-maybe-uninitialized")
endif()

find_package(Sophus 23.0.1 REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)

set(example_source hello_so3 hello_se3)

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

if(${BUILD_FARM_NG_PROTOS})
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
