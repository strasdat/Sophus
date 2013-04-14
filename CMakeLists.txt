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

IF( CMAKE_COMPILER_IS_GNUCXX )
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

   ADD_DEFINITIONS("-Wall -Werror")
ENDIF()

################################################################################
# Add local path for finding packages, set the local version first
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

################################################################################
# Create variables used for exporting in SophusConfig.cmake
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

SET( SOURCES ${SOURCE_DIR}/sophus.hpp ${SOURCE_DIR}/ensure.hpp
             ${SOURCE_DIR}/example_ensure_handler.cpp )

FOREACH(templ ${TEMPLATES})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${templ}.hpp)
ENDFOREACH(templ)

# Add sources as custom target so that they are shown in IDE's
ADD_CUSTOM_TARGET( libsophus SOURCES ${SOURCES} )

################################################################################
# Create 'test' make target using ctest
ENABLE_TESTING()
ADD_SUBDIRECTORY( test )

################################################################################
# Create the SophusConfig.cmake file for other cmake projects.
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/SophusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake @ONLY IMMEDIATE )
EXPORT( PACKAGE Sophus )

INSTALL(DIRECTORY sophus DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING PATTERN "*.hpp" )
