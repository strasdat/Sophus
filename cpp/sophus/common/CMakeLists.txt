#[[
sophus_common



]]
farm_ng_add_library(sophus_common
  NAMESPACE sophus
  INCLUDE_DIR ../../
  HEADERS
  cast.h
  common.h
  concept_utils.h
  enum.h
  point_concepts.h
  point_methods.h
  point_traits.h
  reduce.h
  types.h
)

target_link_libraries(sophus_common INTERFACE Eigen3::Eigen farm_ng_core::farm_ng_core_logging)
