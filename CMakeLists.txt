cmake_minimum_required(VERSION 3.16)
project(Sophus VERSION 2.0.0)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(farm_ng_cmake 0.1.0 REQUIRED)
farm_ng_module()

find_package(Eigen3 3.4.0 REQUIRED)
find_package(farm_ng_core 0.1.0 REQUIRED)


# Determine if sophus is built as a subproject (using add_subdirectory)
# or if it is the master project.
if (NOT DEFINED SOPHUS_MASTER_PROJECT)
  set(SOPHUS_MASTER_PROJECT OFF)
  if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(SOPHUS_MASTER_PROJECT ON)
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif ()
endif ()

if (${COVERAGE})
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} --coverage")
  add_definitions("-DSOPHUS_CODE_COVERAGE")
endif()

option(SOPHUS_INSTALL "Generate the install target." ${SOPHUS_MASTER_PROJECT})
option(SOPHUS_USE_BASIC_LOGGING "Use basic logging (in ensure and test macros)" OFF)

option(SOPHUS_CERES "Enable sophus ceres" ON)

option(BUILD_FARM_NG_PROTOS "Build tests." Off)


option(BUILD_SOPHUS_TESTS "Build tests." ON)
if(${BUILD_SOPHUS_TESTS})
    farm_ng_enable_testing()
endif()


if(${BUILD_FARM_NG_PROTOS})
  find_package(Protobuf REQUIRED)
  include_directories(${Protobuf_INCLUDE_DIRS})
endif()

if(${BUILD_FARM_NG_PROTOS})
  add_subdirectory(protos)
endif()


add_subdirectory(cpp)

farm_ng_export_module(
  NAME Sophus
  REQUIRED_DEPS farm_ng_core fmt)
