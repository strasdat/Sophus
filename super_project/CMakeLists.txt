cmake_minimum_required(VERSION 3.16)
# https://semver.org/
project(farm_ng_all VERSION 0.0.1)


if("${ROW_ACCESS}" STREQUAL "col_major")
    message(STATUS "Testing Sophus with col_major")
    set(EIGEN_DEFAULT_TO_ROW_MAJOR Off)
else()
    message(STATUS "Testing Sophus with row_major")
    set(EIGEN_DEFAULT_TO_ROW_MAJOR On)
endif()


include(FetchContent)
FetchContent_Declare(
  farm_ng_cmake
  GIT_REPOSITORY https://github.com/farm-ng/farm-ng-cmake.git
  GIT_TAG 031979bdca37cd8777674ce1329ff7dcc612d929
)
FetchContent_MakeAvailable(farm_ng_cmake)
set(farm_ng_cmake_DIR ${farm_ng_cmake_SOURCE_DIR}/cmake)
set(BUILD_SHARED_LIBS ON)
find_package(farm_ng_cmake REQUIRED)

include(External_fmt)
include(External_expected)
include(External_eigen)
include(External_ceres)

ExternalProject_Add(farm-ng-core
    DEPENDS fmt expected eigen
    GIT_REPOSITORY  "https://github.com/farm-ng/farm-ng-core.git"
    GIT_TAG "c90f2b7cf6cd87e7de7d679421c1bfd636c71279"
    PREFIX ${farm_ng_EXT_PREFIX}
    CMAKE_ARGS
    ${farm_ng_DEFAULT_ARGS}
    -Dfarm_ng_cmake_DIR=${farm_ng_cmake_DIR}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DBUILD_FARM_NG_PROTOS=On
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    )

set(SUPER_PROJ_FARM_NG_PROTOS ${BUILD_PROTOS})

ExternalProject_Add(Sophus
    DEPENDS farm-ng-core eigen ceres
    PREFIX ${farm_ng_EXT_PREFIX}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/Sophus-build
    CMAKE_ARGS ${farm_ng_DEFAULT_ARGS}
    -DBUILD_FARM_NG_PROTOS=${SUPER_PROJ_FARM_NG_PROTOS}
    -DBUILD_SOPHUS_TESTS=ON
    -Dfarm_ng_cmake_DIR=${farm_ng_cmake_DIR}
    TEST_BEFORE_INSTALL ON
)

ExternalProject_Add(SophusTestInstallTargets
    DEPENDS Sophus
    PREFIX ${farm_ng_EXT_PREFIX}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/examples/test_install_targets
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/SophusTestInstallTargets-build
    CMAKE_ARGS ${farm_ng_DEFAULT_ARGS}
    -DBUILD_FARM_NG_PROTOS=${SUPER_PROJ_FARM_NG_PROTOS}
    INSTALL_COMMAND ""
)
