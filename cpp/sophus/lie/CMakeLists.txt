#[[
sophus_lie



]]

set(sophus_lie_src_prefixes
   group_manifold
   identity
   isometry2
   isometry3
   lie_group
   rotation2
   rotation3
   scaling_translation
   scaling
   similarity2
   similarity3
   spiral_similarity2
   spiral_similarity3
   translation
)

set(sophus_lie_h)

foreach(x ${sophus_lie_src_prefixes} )
    list(APPEND sophus_lie_h ${x}.h)
endforeach()



farm_ng_add_library(sophus_lie
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  ${sophus_lie_h}
)
target_link_libraries(sophus_lie INTERFACE sophus_linalg)


if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_lie_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_lie
                          LINK_LIBRARIES sophus_lie
                          LABELS small)
  endforeach()
endif()

if(${BUILD_FARM_NG_PROTOS})
    add_subdirectory(proto)
endif()

add_subdirectory(interp)
