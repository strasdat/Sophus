#[[
sophus_ceres



]]
farm_ng_add_library(sophus_ceres
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  jet_helpers.h
  manifold.h
  typetraits.h
)
target_link_libraries(sophus_ceres INTERFACE sophus_lie)


if(${BUILD_SOPHUS_TESTS})
  if(${SOPHUS_CERES})
    find_package(Ceres 2.1.0 REQUIRED)
    foreach(test_basename
      ceres
      manifold
    )
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_ceres
                          LINK_LIBRARIES
                          Ceres::ceres sophus_ceres
                          farm_ng_core::farm_ng_core_pipeline
                          sophus_sensor
                          LABELS large)
    target_compile_definitions(${test_basename}_test_sophus_ceres PRIVATE ${ARGN})
    endforeach()
  endif()
endif()
