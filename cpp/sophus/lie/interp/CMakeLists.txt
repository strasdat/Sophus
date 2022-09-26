#[[
sophus_lie_interp



]]

farm_ng_add_library(sophus_lie_interp
  NAMESPACE sophus
  INCLUDE_DIR ../../../
  HEADERS
  average.h
  interpolate.h
  spline.h
  details/interpolate_impl.h
)
target_link_libraries(sophus_lie_interp INTERFACE sophus_lie)

