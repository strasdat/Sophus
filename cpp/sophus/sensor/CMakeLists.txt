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

add_library(sophus_sensor SHARED
  ${sophus_sensor_cpp}
  ${sophus_sensor_h})
target_link_libraries(sophus_sensor sophus_lie sophus_image)
add_library (sophus::sophus_sensor ALIAS sophus_sensor)


target_sources(sophus_sensor INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES
                  ${sophus_sensor_h}
                  camera_transforms/affine.h
                  camera_transforms/brown_conrady.h
                  camera_transforms/kannala_brandt.h
                  camera_transforms/orthographic.h
               )

if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename ${sophus_sensor_src_prefixes})
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_sensor
                          LINK_LIBRARIES
                          LABELS small)
  endforeach()
endif()

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()
