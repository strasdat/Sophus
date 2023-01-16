#[[
sophus_calculus



]]

farm_ng_add_library(sophus_calculus
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  num_diff.h
  region.h
)
target_link_libraries(sophus_calculus INTERFACE sophus_common)

if(${BUILD_SOPHUS_TESTS})
  farm_ng_add_test(region
    PARENT_LIBRARY sophus_calculus
    LABELS small)
endif()
