#[[
sophus_sensor



]]

set(sophus_sensor_src_prefixes
    camera_model
    clipping_planes
    imu_model
    camera_rig
    )

set(sophus_sensor_cpp)
set(sophus_sensor_h)
foreach(x ${sophus_sensor_src_prefixes} )
    list(APPEND sophus_sensor_cpp ${x}.cpp)
    list(APPEND sophus_sensor_h ${x}.h)
endforeach()

farm_ng_add_library(sophus_sensor SHARED
NAMESPACE sophus
INCLUDE_DIR ../../
HEADERS
${sophus_sensor_h}
camera_distortion/affine.h
camera_distortion/brown_conrady.h
camera_distortion/kannala_brandt.h
camera_distortion/orthographic.h
SOURCES
  ${sophus_sensor_cpp})

target_link_libraries(sophus_sensor sophus_lie sophus_image)


if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_sensor_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_sensor
                          LINK_LIBRARIES sophus_sensor
                          LABELS small)
  endforeach()
endif()

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()
