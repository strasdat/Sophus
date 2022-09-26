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
farm_ng_add_library(sophus_experimental 
NAMESPACE sophus
  INCLUDE_DIR ../../
  HEADERS
  ${sophus_experimental_h}
  SOURCES
  ${sophus_experimental_cpp}
)
target_link_libraries(sophus_experimental sophus_sensor sophus_image)

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_experimental_src_prefixes})
    farm_ng_add_test(${test_basename}
                        PARENT_LIBRARY sophus_experimental
                        LINK_LIBRARIES sophus_experimental
                        LABELS small)
  endforeach()
endif()
