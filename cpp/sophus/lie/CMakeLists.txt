#[[
sophus_lie



]]

set(sophus_lie_src_prefixes
  cartesian
  rxso2
  rxso3
  se2
  se3
  sim2
  sim3
  so2
  so3
  details/sim_impl
)

set(sophus_lie_h)
foreach(x ${sophus_lie_src_prefixes} )
    list(APPEND sophus_lie_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_lie INTERFACE
  ${sophus_lie_h}
)
target_link_libraries(sophus_lie INTERFACE sophus_linalg)
add_library (sophus::sophus_lie ALIAS sophus_lie)

target_sources(sophus_lie INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_lie_h})


set(sophus_lie_test_prefixes
  cartesian2
  cartesian3
  common
  rxso2
  rxso3
  se2
  se3
  sim2
  sim3
  so2
  so3
)

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_lie_test_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_lie
                          LINK_LIBRARIES
                          LABELS small)
  endforeach()
endif()

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()

add_subdirectory(interp)
