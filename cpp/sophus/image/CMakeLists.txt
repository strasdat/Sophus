#[[
sophus_image



]]

set(sophus_image_src_prefixes
  interpolation
  image
  image_size
  image_types
  image_view
  mut_image
  mut_image_view
  mut_runtime_image_view
  runtime_image
  runtime_image_view
  runtime_image_types
)

set(sophus_image_cpp)
set(sophus_image_h)
foreach(x ${sophus_image_src_prefixes} )
    list(APPEND sophus_image_cpp ${x}.cpp)
    list(APPEND sophus_image_h ${x}.h)
endforeach()

farm_ng_add_library(sophus_image
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  ${sophus_image_h}
  SOURCES
  ${sophus_image_cpp}
)
target_link_libraries(sophus_image sophus_geometry)



if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_image_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_image
                          LINK_LIBRARIES sophus_image
                          LABELS small)
  endforeach()
endif()

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()
