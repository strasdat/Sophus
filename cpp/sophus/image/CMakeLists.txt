#[[
sophus_image



]]

set(sophus_image_src_prefixes
  interpolation
  image
  image_size
  image_types
  image_view
  runtime_image
)

set(sophus_image_cpp)
set(sophus_image_h)
foreach(x ${sophus_image_src_prefixes} )
    list(APPEND sophus_image_cpp ${x}.cpp)
    list(APPEND sophus_image_h ${x}.h)
endforeach()

add_library(sophus_image SHARED
  ${sophus_image_cpp}
  ${sophus_image_h})
target_link_libraries(sophus_image sophus_geometry)
add_library (sophus::sophus_image ALIAS sophus_image)

target_sources(sophus_image INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_image_h})


if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_image_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_image
                          LINK_LIBRARIES
                          LABELS small)
  endforeach()
endif()
