#[[
sophus_manifold



]]

set(sophus_manifold_src_prefixes
    complex
    product_manifold
    quaternion
    unit_vector
)

set(sophus_manifold_h)

foreach(x ${sophus_manifold_src_prefixes} )
    list(APPEND sophus_manifold_h ${x}.h)
endforeach()

farm_ng_add_library(sophus_manifold
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  ${sophus_manifold_h}
)

target_link_libraries(sophus_manifold INTERFACE sophus_linalg)

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_manifold_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_manifold
                          LINK_LIBRARIES sophus_manifold
                          LABELS small)
  endforeach()
endif()
