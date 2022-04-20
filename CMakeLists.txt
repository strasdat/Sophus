cmake_minimum_required(VERSION 3.4)
project(Sophus VERSION 22.04.1)

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

  set(CMAKE_CXX_STANDARD 14)

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

if(SOPHUS_USE_BASIC_LOGGING)
  set (CMAKE_DISABLE_FIND_PACKAGE_fmt ON)
endif()

# Find public dependencies if targets are not yet defined. (Targets might be for example
# defined by a parent project including Sophus via `add_subdirectory`.)

if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 3.3.0 REQUIRED)
endif()

if(NOT TARGET fmt::fmt)
  find_package(fmt)
endif()


# Define interface library target
add_library(sophus INTERFACE)
add_library (Sophus::Sophus ALIAS sophus)

set(SOPHUS_HEADER_FILES
  sophus/average.hpp
  sophus/cartesian.hpp
  sophus/ceres_local_parameterization.hpp
  sophus/common.hpp
  sophus/geometry.hpp
  sophus/interpolate.hpp
  sophus/interpolate_details.hpp
  sophus/num_diff.hpp
  sophus/rotation_matrix.hpp
  sophus/rxso2.hpp
  sophus/rxso3.hpp
  sophus/se2.hpp
  sophus/se3.hpp
  sophus/sim2.hpp
  sophus/sim3.hpp
  sophus/sim_details.hpp
  sophus/so2.hpp
  sophus/so3.hpp
  sophus/spline.hpp
  sophus/types.hpp
  sophus/velocities.hpp
)

set(SOPHUS_OTHER_FILES
  sophus/test_macros.hpp
  sophus/example_ensure_handler.cpp
)

if(MSVC)
  # Define common math constants if we compile with MSVC
  target_compile_definitions (sophus INTERFACE _USE_MATH_DEFINES)
endif (MSVC)

# Add Eigen interface dependency, depending on available cmake info
if(TARGET Eigen3::Eigen)
  target_link_libraries(sophus INTERFACE Eigen3::Eigen)
  set(Eigen3_DEPENDENCY "find_dependency (Eigen3 ${Eigen3_VERSION})")
else()
  target_include_directories (sophus SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})
endif()

if(SOPHUS_USE_BASIC_LOGGING OR NOT TARGET fmt::fmt)
  # NOTE fmt_FOUND does not seem to be defined even though the package config
  # was found.
  target_compile_definitions(sophus INTERFACE SOPHUS_USE_BASIC_LOGGING=1)
  message(STATUS "Turning basic logging ON")
else()
  target_link_libraries(sophus INTERFACE fmt::fmt)
  set(fmt_DEPENDENCY "find_dependency (fmt ${fmt_VERSION})")
  message(STATUS "Turning basic logging OFF")
endif()

# Associate target with include directory
target_include_directories(sophus INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Declare all used C++14 features
target_compile_features (sophus INTERFACE
  cxx_auto_type
  cxx_decltype
  cxx_nullptr
  cxx_right_angle_brackets
  cxx_variadic_macros
  cxx_variadic_templates
)

# Add sources as custom target so that they are shown in IDE's
add_custom_target(other SOURCES ${SOPHUS_OTHER_FILES} ${SOPHUS_HEADER_FILES})

# Create 'test' make target using ctest
option(BUILD_SOPHUS_TESTS "Build tests." ON)
if(BUILD_SOPHUS_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Create examples make targets using ctest
option(BUILD_SOPHUS_EXAMPLES "Build examples." ON)
if(BUILD_SOPHUS_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SOPHUS_INSTALL)
  # Export package for use from the build tree
  set(SOPHUS_CMAKE_EXPORT_DIR ${CMAKE_INSTALL_DATADIR}/sophus/cmake)

  set_target_properties(sophus PROPERTIES EXPORT_NAME Sophus)

  install(TARGETS sophus EXPORT SophusTargets)
  install(EXPORT SophusTargets
    NAMESPACE Sophus::
    DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
  )

  export(TARGETS sophus NAMESPACE Sophus:: FILE SophusTargets.cmake)
  export(PACKAGE Sophus)

  configure_package_config_file(
    SophusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake
    INSTALL_DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  # Remove architecture dependence. Sophus is a header-only library.
  set(TEMP_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
  unset(CMAKE_SIZEOF_VOID_P)

  # Write version to file
  write_basic_package_version_file (
    SophusConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  # Recover architecture dependence
  set(CMAKE_SIZEOF_VOID_P ${TEMP_SIZEOF_VOID_P})

  # Install cmake targets
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/SophusConfigVersion.cmake
    DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
  )

  # Install header files
  install(
    FILES ${SOPHUS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sophus
  )
endif()
