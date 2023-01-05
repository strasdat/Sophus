#[[
sophus_calculus



]]

farm_ng_add_library(sophus_calculus
  NAMESPACE sophus
  INCLUDE_DIR ../../
  HEADERS
  interval.h
  num_diff.h
)
target_link_libraries(sophus_calculus INTERFACE sophus_common)

if(${BUILD_SOPHUS_TESTS})
  farm_ng_add_test(interval
    PARENT_LIBRARY sophus_calculus
    LABELS small)
endif()
