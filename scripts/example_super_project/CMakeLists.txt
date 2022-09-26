# Place this file in a directory with:
# - farm-ng-core # branch cmake-super-project
# - Sophus # branch cmake-super-project
#
# Then:
#  mkdir build
#  cd build
#  cmake ..
#  make

cmake_minimum_required(VERSION 3.16)
# https://semver.org/
project(farm_ng_all VERSION 0.0.1)


list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/farm-ng-core/cmake)

set(BUILD_SHARED_LIBS ON)
include(farm_ng_default_args)

include(External_fmt)
include(External_expected)
include(External_eigen)
include(External_ceres)
include(External_subdir_farm-ng-core)
include(External_subdir_Sophus)



ExternalProject_Add(farm-ng-core-TestInstallTargets
    DEPENDS farm-ng-core
    PREFIX ${farm_ng_EXT_PREFIX}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/farm-ng-core/cpp/examples/test_install_targets
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/farm-ng-core-TestInstallTargets-build
    CMAKE_ARGS ${farm_ng_DEFAULT_ARGS}
    -DBUILD_FARM_NG_PROTOS=ON
    INSTALL_COMMAND ""
)

ExternalProject_Add(SophusTestInstallTargets
    DEPENDS Sophus
    PREFIX ${farm_ng_EXT_PREFIX}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Sophus/cpp/examples/test_install_targets
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/SophusTestInstallTargets-build
    CMAKE_ARGS ${farm_ng_DEFAULT_ARGS}
    -DBUILD_FARM_NG_PROTOS=ON
    INSTALL_COMMAND ""
)
