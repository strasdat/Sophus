#  mkdir build
#  cd build
#  cmake ..
#  make

cmake_minimum_required(VERSION 3.16)
# https://semver.org/
project(farm_ng_all VERSION 0.0.1)


list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(BUILD_SHARED_LIBS ON)
include(farm_ng_default_args)

include(External_fmt)
include(External_expected)
include(External_eigen)
include(External_ceres)
include(External_farm-ng-core)

ExternalProject_Add(Sophus
    DEPENDS farm-ng-core eigen ceres
    PREFIX ${farm_ng_EXT_PREFIX}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/Sophus-build
    CMAKE_ARGS ${farm_ng_DEFAULT_ARGS}
    -DBUILD_FARM_NG_PROTOS=On
    -DBUILD_SOPHUS_TESTS=ON
    TEST_BEFORE_INSTALL ON
)


ExternalProject_Add(SophusTestInstallTargets
    DEPENDS Sophus
    PREFIX ${farm_ng_EXT_PREFIX}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/examples/test_install_targets
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/SophusTestInstallTargets-build
    CMAKE_ARGS ${farm_ng_DEFAULT_ARGS}
    -DBUILD_FARM_NG_PROTOS=ON
    INSTALL_COMMAND ""
)
