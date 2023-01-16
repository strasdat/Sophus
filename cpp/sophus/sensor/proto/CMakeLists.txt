farm_ng_add_library(sophus_sensor_proto_conv
NAMESPACE Sophus
INCLUDE_DIR ../../..
HEADERS
  conv.h
SOURCES
  conv.cpp
)

target_link_libraries(sophus_sensor_proto_conv PUBLIC
sophus_sensor
protobuf::libprotobuf
farm_ng_core::farm_ng_core_prototools
sophus_sensor_proto_defs
sophus_image_proto_conv
)

if(${BUILD_SOPHUS_TESTS})
farm_ng_add_test(conv
              PARENT_LIBRARY sophus_sensor_proto_conv
              LINK_LIBRARIES sophus_sensor_proto_conv
              LABELS small)
endif()
