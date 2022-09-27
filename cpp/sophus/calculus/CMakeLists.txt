#[[
sophus_calculus



]]

farm_ng_add_library(sophus_calculus
  NAMESPACE sophus
  INCLUDE_DIR ../../
  HEADERS
  num_diff.h
)
target_link_libraries(sophus_calculus INTERFACE sophus_common)
