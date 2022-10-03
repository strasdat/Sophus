#[[
sophus_ceres



]]
farm_ng_add_library(sophus_ceres
  NAMESPACE sophus
  INCLUDE_DIR ../../
  HEADERS
  jet_helpers.h
  manifold.h
  typetraits.h
)
target_link_libraries(sophus_ceres INTERFACE sophus_lie)


find_package(Ceres 2.1.0 REQUIRED)
foreach(test_basename
    rxso2
    rxso3
    se2
    se3
    sim2
    sim3
    so2
    so3)
  farm_ng_add_test(${test_basename}
                        PARENT_LIBRARY sophus_ceres
                        LINK_LIBRARIES
                        Ceres::ceres sophus_ceres
                        LABELS large)
  target_compile_definitions(${test_basename}_test_sophus_ceres PRIVATE ${ARGN})
endforeach()
