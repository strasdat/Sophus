# git clone https://ceres-solver.googlesource.com/ceres-solver
find_package( CeresSolver QUIET )

if( CeresSolver_FOUND )
  # Ensure that ${Sophus_INCLUDE_DIR} is first on search path
  INCLUDE_DIRECTORIES( BEFORE ${Sophus_INCLUDE_DIR} ${CeresSolver_INCLUDE_DIRS})

  MESSAGE(STATUS "CERES found")

  # Tests to run
  SET( TEST_SOURCES test_ceres_se3 )

  FOREACH(test_src ${TEST_SOURCES})
    ADD_EXECUTABLE( ${test_src} ${test_src}.cpp)
    TARGET_LINK_LIBRARIES( ${test_src} ${CeresSolver_LIBRARIES} )
    ADD_TEST( ${test_src} ${test_src} )
  ENDFOREACH(test_src)

endif( CeresSolver_FOUND )
