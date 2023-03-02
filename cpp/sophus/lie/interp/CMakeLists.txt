#[[
sophus_lie_interp



]]

farm_ng_add_library(sophus_lie_interp
  NAMESPACE Sophus
  INCLUDE_DIR ../../../
  HEADERS
  average.h
  interpolate.h
  spline.h
  details/interpolate_impl.h
)
target_link_libraries(sophus_lie_interp INTERFACE sophus_lie)

farm_ng_add_test(interpolate
                      PARENT_LIBRARY sophus_lie_interp
                      LINK_LIBRARIES sophus_lie_interp
                      LABELS small)
