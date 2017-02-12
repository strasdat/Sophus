SET(PROJECT_NAME Sophus)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET( CMAKE_VERBOSE_MAKEFILE ON)

################################################################################
# Release by default
# Turn on Debug with "-DCMAKE_BUILD_TYPE=Debug"
IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

set(CMAKE_CXX_STANDARD 14)
IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
   SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -Wno-deprecated-register -std=c++14 -stdlib=libc++ -Wno-deprecated-register")
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
   SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -std=c++14 -Wno-deprecated-declarations -ftemplate-backtrace-limit=0")
   SET(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline")
   SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
   SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
   ADD_DEFINITIONS("-D _USE_MATH_DEFINES /wd4305 /wd4244")
ENDIF()

################################################################################
# Add local path for finding packages, set the local version first
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

################################################################################
# Create variables used for exporting (from build tree) in SophusConfig.cmake
SET( Sophus_LIBRARIES "" )
SET( Sophus_INCLUDE_DIR ${PROJECT_SOURCE_DIR} )

################################################################################
FIND_PACKAGE( Eigen3 REQUIRED )
LIST( APPEND Sophus_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} )

################################################################################
SET( SOURCE_DIR "sophus")

SET( TEMPLATES so2
               se2
               so3
               se3
               rxso3
               sim3
)

SET( SOURCES ${SOURCE_DIR}/common.hpp
             ${SOURCE_DIR}/types.hpp
             ${SOURCE_DIR}/example_ensure_handler.cpp )

FOREACH(templ ${TEMPLATES})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${templ}.hpp)
ENDFOREACH(templ)

# Add sources as custom target so that they are shown in IDE's
ADD_CUSTOM_TARGET( libsophus SOURCES ${SOURCES} )

################################################################################
# Create 'test' make target using ctest
option(BUILD_TESTS "Build tests." ON)
if(BUILD_TESTS)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY( test )
endif()

################################################################################
# Export package for use from the build tree
EXPORT( PACKAGE Sophus )

# Create the SophusConfig.cmake file for other cmake projects.
# ... for the build tree
SET( CONFIG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET( CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR})
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/SophusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake @ONLY )
# ... for the install tree
SET( CMAKECONFIG_INSTALL_DIR lib/cmake/Sophus )
FILE( RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
    "${CMAKE_INSTALL_PREFIX}/include" )

SET( Sophus_INCLUDE_DIR "\${Sophus_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
SET( CONFIG_SOURCE_DIR )
SET( CONFIG_DIR )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/SophusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SophusConfig.cmake @ONLY )

INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SophusConfig.cmake"
        DESTINATION ${CMAKECONFIG_INSTALL_DIR} )

# Install headers
INSTALL(DIRECTORY sophus DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING PATTERN "*.hpp" )
