#[[
sophus_geometry



]]

set(sophus_geometry_src_prefixes
  fit_plane
  inverse_depth
  plane_conv
  point_transform
  projection
  ray
)

set(sophus_geometry_h)
foreach(x ${sophus_geometry_src_prefixes} )
    list(APPEND sophus_geometry_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_geometry INTERFACE
  ${sophus_geometry_h}
)
target_link_libraries(sophus_geometry INTERFACE sophus_lie)
add_library (sophus::sophus_geometry ALIAS sophus_geometry)

target_sources(sophus_geometry INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_geometry_h})


 if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_geometry_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_geometry
                          LINK_LIBRARIES
                          LABELS small)
  endforeach()
endif()

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()
