# Make sure Ceres knows where to find Eigen
list(APPEND SEARCH_HEADERS ${EIGEN3_INCLUDE_DIR})

# git clone https://ceres-solver.googlesource.com/ceres-solver
find_package( Ceres 2.0.0 QUIET )

if( Ceres_FOUND )
  MESSAGE(STATUS "CERES found")

  # Tests to run
  SET( TEST_SOURCES test_ceres_se3 test_ceres_so3 test_ceres_sim3 test_ceres_rxso3 test_ceres_se2 test_ceres_so2 test_ceres_rxso2 test_ceres_sim2 )

  FOREACH(test_src ${TEST_SOURCES})
    ADD_EXECUTABLE( ${test_src} ${test_src}.cpp local_parameterization_se3.hpp)
    TARGET_LINK_LIBRARIES( ${test_src} sophus Ceres::ceres)
    ADD_TEST( ${test_src} ${test_src} )
  ENDFOREACH(test_src)

endif( Ceres_FOUND )
