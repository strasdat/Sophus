#[[
sophus_common



]]
farm_ng_add_library(sophus_concept
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  image.h
  lie_group.h
  lie_group_prop_tests.h
  manifold.h
  manifold_prop_tests.h
  params.h
  point.h
  utils.h
)

target_link_libraries(sophus_concept INTERFACE sophus_common)

foreach(test_basename
  group_accessors
  lie_group
  manifold
  params
  point
  utils)

  farm_ng_add_test(${test_basename}
                   PARENT_LIBRARY sophus_concept
                   LINK_LIBRARIES sophus_concept
                   LABELS small)
endforeach()
