#[[
sophus_common



]]
farm_ng_add_library(sophus_common
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  common.h
  enum.h
)

farm_ng_add_test(common
  PARENT_LIBRARY sophus_common
  LINK_LIBRARIES sophus_common
  LABELS small)

target_link_libraries(
  sophus_common
  INTERFACE
    Eigen3::Eigen
    farm_ng_core::farm_ng_core_logging
  )
