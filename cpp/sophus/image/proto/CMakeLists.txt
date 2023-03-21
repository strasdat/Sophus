farm_ng_add_library(sophus_image_proto_conv
  NAMESPACE Sophus
  INCLUDE_DIR ../../..
  HEADERS
    conv.h
  SOURCES
    conv.cpp
)

target_link_libraries(sophus_image_proto_conv PUBLIC
  sophus_linalg
  protobuf::libprotobuf
  farm_ng_core::farm_ng_core_proto_defs
  farm_ng_core::farm_ng_core_prototools
  sophus_image_proto_defs
  sophus_image
 )

if(${BUILD_SOPHUS_TESTS})
  farm_ng_add_test(conv
                PARENT_LIBRARY sophus_image_proto_conv
                LINK_LIBRARIES sophus_image_proto_conv
                LABELS small)
endif()
