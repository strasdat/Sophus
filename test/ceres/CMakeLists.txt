# Make sure Ceres knows where to find Eigen
list(APPEND SEARCH_HEADERS ${EIGEN3_INCLUDE_DIR})

# git clone https://ceres-solver.googlesource.com/ceres-solver
find_package( Ceres 1.6.0 QUIET )

if( Ceres_FOUND )
  # Ensure that ${Sophus_INCLUDE_DIR} is first on search path
  INCLUDE_DIRECTORIES( BEFORE ${Sophus_INCLUDE_DIR} ${CERES_INCLUDES})

  MESSAGE(STATUS "CERES found")

  # Tests to run
  SET( TEST_SOURCES test_ceres_se3 )

  FOREACH(test_src ${TEST_SOURCES})
    ADD_EXECUTABLE( ${test_src} ${test_src}.cpp local_parameterization_se3)
    TARGET_LINK_LIBRARIES( ${test_src} ${CERES_LIBRARIES} )
    ADD_TEST( ${test_src} ${test_src} )
  ENDFOREACH(test_src)

endif( Ceres_FOUND )
