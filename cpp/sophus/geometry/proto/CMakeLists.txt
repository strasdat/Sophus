farm_ng_add_library(sophus_geometry_proto_conv
  NAMESPACE Sophus
  INCLUDE_DIR ../../..
  HEADERS
    conv.h
  SOURCES
    conv.cpp
)

target_link_libraries(sophus_geometry_proto_conv PUBLIC
  sophus_lie
  protobuf::libprotobuf
  farm_ng_core::farm_ng_core_prototools
  sophus_geometry_proto_defs
  sophus_linalg_proto_conv)

if(${BUILD_SOPHUS_TESTS})
  farm_ng_add_test(conv
                PARENT_LIBRARY sophus_geometry_proto_conv
                LINK_LIBRARIES sophus_geometry_proto_conv
                LABELS small)
endif()
