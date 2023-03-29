#[[
sophus_linalg



]]

set(sophus_linalg_src_prefixes
    cast
    homogeneous
    orthogonal
    point_traits
    reduce
    vector_space
    vector_space_traits
)

set(sophus_linalg_h)

foreach(x ${sophus_linalg_src_prefixes} )
    list(APPEND sophus_linalg_h ${x}.h)
endforeach()

farm_ng_add_library(sophus_linalg
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  ${sophus_linalg_h}
)

target_link_libraries(sophus_linalg INTERFACE sophus_common sophus_concept)

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_linalg_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_linalg
                          LINK_LIBRARIES sophus_linalg
                          LABELS small)
  endforeach()
endif()


if(${BUILD_FARM_NG_PROTOS})
    add_subdirectory(proto)
endif()
