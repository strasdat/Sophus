cmake_minimum_required(VERSION 3.24)
project(Sophus VERSION 22.99.0)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Determine if sophus is built as a subproject (using add_subdirectory)
# or if it is the master project.
if (NOT DEFINED SOPHUS_MASTER_PROJECT)
  set(SOPHUS_MASTER_PROJECT OFF)
  if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(SOPHUS_MASTER_PROJECT ON)
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif ()
endif ()

option(SOPHUS_INSTALL "Generate the install target." ${SOPHUS_MASTER_PROJECT})
option(SOPHUS_USE_BASIC_LOGGING "Use basic logging (in ensure and test macros)" OFF)

if(SOPHUS_MASTER_PROJECT)
  # Release by default
  # Turn on Debug with "-DCMAKE_BUILD_TYPE=Debug"
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  set(CMAKE_CXX_STANDARD 17)

  # Set compiler specific settings (FixMe: Should not cmake do this for us automatically?)
  IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra  -Wno-deprecated-register -Qunused-arguments -fcolor-diagnostics")
  ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -std=c++14 -Wno-deprecated-declarations -ftemplate-backtrace-limit=0")
    SET(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline")
    SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
    SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
  ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "^MSVC$")
    ADD_DEFINITIONS("-D _USE_MATH_DEFINES /bigobj /wd4305 /wd4244 /MP")
  ENDIF()

  # Add local path for finding packages, set the local version first
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
endif()

find_package(Eigen3 3.4.0 REQUIRED)

option(BUILD_SOPHUS_TESTS "Build tests." OFF)
if(${BUILD_SOPHUS_TESTS})
    enable_testing()
endif()

find_package(GTest REQUIRED)

add_subdirectory(thirdparty/farm-ng-core)

add_subdirectory(sophus)

add_library(sophus INTERFACE)
target_link_libraries(sophus INTERFACE
  sophus_core
  sophus_math
  sophus_lie
  sophus_geometry
  sophus_interp
)
add_library (sophus::sophus ALIAS sophus)

# Associate target with include directory
target_include_directories(sophus INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Create examples make targets using ctest
option(BUILD_SOPHUS_EXAMPLES "Build examples." ON)

install(
  TARGETS
     sophus_core
     sophus_math
     sophus_lie
     sophus_geometry
     sophus_interp
     sophus_image
     sophus_sensor
     sophus_ceres
     sophus_experimental
  EXPORT SophusTargets
  FILE_SET HEADERS
  LIBRARY DESTINATION lib
)

write_basic_package_version_file(
    SophusConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(EXPORT SophusTargets
    FILE SophusTargets.cmake
    NAMESPACE sophus::
    DESTINATION share/sophus/cmake
)

configure_file(cmake/SophusConfig.cmake.in SophusConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/SophusConfigVersion.cmake"
        DESTINATION share/sophus/cmake)
