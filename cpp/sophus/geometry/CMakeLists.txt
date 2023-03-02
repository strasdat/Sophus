#[[
sophus_geometry



]]

farm_ng_add_library(sophus_geometry
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  fit_plane.h
  inverse_depth.h
  plane_conv.h
  point_transform.h
  ray.h
)
target_link_libraries(sophus_geometry INTERFACE sophus_lie)

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename
    fit_plane
    inverse_depth
    plane_conv
    point_transform
    ray)
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_geometry
                          LINK_LIBRARIES sophus_geometry
                          LABELS small)
  endforeach()
endif()

if(${BUILD_FARM_NG_PROTOS})
    add_subdirectory(proto)
endif()
