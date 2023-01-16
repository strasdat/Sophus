#[[
sophus_lie



]]

farm_ng_add_library(sophus_lie
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  cartesian.h
  rxso2.h
  rxso3.h
  se2.h
  se3.h
  sim2.h
  sim3.h
  so2.h
  so3.h
  details/sim_impl.h
)
target_link_libraries(sophus_lie INTERFACE sophus_linalg)


if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename
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
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_lie
                          LINK_LIBRARIES sophus_lie
                          LABELS small)
  endforeach()
endif()

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()

add_subdirectory(interp)
