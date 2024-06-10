# Make sure Ceres knows where to find Eigen
list(APPEND SEARCH_HEADERS ${EIGEN3_INCLUDE_DIR})

# git clone https://ceres-solver.googlesource.com/ceres-solver
find_package(Ceres 2)

function(add_test_ceres source postfix)
  add_executable(${source}_${postfix} ${source}.cpp)
  target_link_libraries(${source}_${postfix} sophus Ceres::ceres)
  target_compile_definitions(${source}_${postfix} PRIVATE ${ARGN})
  add_test(${source}_${postfix} ${source}_${postfix})
endfunction()

if(Ceres_FOUND)
  message(STATUS "CERES found")

  # Tests to run
  set(TEST_SOURCES
      test_ceres_so3
      test_ceres_rxso3
      test_ceres_se3
      test_ceres_sim3
      test_ceres_so2
      test_ceres_rxso2
      test_ceres_se2
      test_ceres_sim2)

  foreach(test_src ${TEST_SOURCES})
    add_test_ceres(${test_src} "manifold")
  endforeach()
endif(Ceres_FOUND)
