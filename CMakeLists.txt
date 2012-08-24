SET(PROJECT_NAME Sophus)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET (CMAKE_VERBOSE_MAKEFILE ON)

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

IF (CMAKE_COMPILER_IS_GNUCXX )
   SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ")

  ADD_DEFINITIONS("-Wall -Werror -Wno-unused-variable
                   -Wno-unused-but-set-variable -Wno-unknown-pragmas ")
ENDIF()

################################################################################
# Add local path for finding packages, set the local version first
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" )
list( APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules" )

################################################################################
# Create variables used for exporting in SophusConfig.cmake
set( Sophus_LIBRARIES "" )
set( Sophus_INCLUDE_DIR ${PROJECT_SOURCE_DIR} )

################################################################################

#SET (INCLUDE_DIRS "../eigen3.1/")
find_package( Eigen3 REQUIRED )
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )
SET( Sophus_INCLUDE_DIR ${Sophus_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} )

SET (SOURCE_DIR "sophus")
SET (CLASSES  so2
              se2
              se3
              so3
              scso3
              sim3
)

SET (SOURCES)

FOREACH(class ${CLASSES})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${class}.cpp ${SOURCE_DIR}/${class}.h)
ENDFOREACH(class)

LINK_LIBRARIES (${PROJECT_NAME} ${LIBS})

set( Sophus_LIBRARIES ${Sophus_LIBRARIES} ${LIBS} )

INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})

ADD_EXECUTABLE(test_so2 sophus/test_so2.cpp)
ADD_EXECUTABLE(test_se2 sophus/test_se2.cpp)
ADD_EXECUTABLE(test_so3 sophus/test_so3.cpp)
ADD_EXECUTABLE(test_scso3 sophus/test_scso3.cpp)
ADD_EXECUTABLE(test_se3 sophus/test_se3.cpp)
ADD_EXECUTABLE(test_sim3 sophus/test_sim3.cpp)
ENABLE_TESTING()

ADD_TEST(test_so2 test_so2)
ADD_TEST(test_se2 test_se2)
ADD_TEST(test_so3 test_so3)
ADD_TEST(test_scso3 test_scso3)
ADD_TEST(test_se3 test_se3)
ADD_TEST(test_sim3 test_sim3)

##############################################################################
# Get full library name
GET_TARGET_PROPERTY( FULL_LIBRARY_NAME ${PROJECT_NAME} LOCATION )
set( Sophus_LIBRARIES ${Sophus_LIBRARIES} ${FULL_LIBRARY_NAME} )
set( Sophus_LIBRARY_DIR ${PROJECT_BINARY_DIR} )

################################################################################
# Create the SophusConfig.cmake file for other cmake projects.
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/SophusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake @ONLY IMMEDIATE )
export( PACKAGE Sophus )

INSTALL(DIRECTORY sophus DESTINATION ${CMAKE_INSTALL_PREFIX}/include FILES_MATCHING PATTERN "*.h" )
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )