#[[
sophus_experimental



]]

set(sophus_experimental_src_prefixes
    least_squares
    )

set(sophus_experimental_cpp)
set(sophus_experimental_h)
foreach(x ${sophus_experimental_src_prefixes} )
    list(APPEND sophus_experimental_cpp ${x}.cpp)
    list(APPEND sophus_experimental_h ${x}.h)
endforeach()
add_library(sophus_experimental SHARED
  ${sophus_experimental_cpp}
  ${sophus_experimental_h})
target_link_libraries(sophus_experimental sophus_sensor sophus_image)

target_sources(sophus_image INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_experimental_h})

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_experimental_src_prefixes})
    farm_ng_add_test(${test_basename}
                        PARENT_LIBRARY sophus_experimental
                        LINK_LIBRARIES
                        LABELS small)
  endforeach()
endif()
