farm_ng_add_library(sophus_linalg_proto_conv
  NAMESPACE Sophus
  INCLUDE_DIR ../../..
  HEADERS
    conv.h
  SOURCES
    conv.cpp
)

target_link_libraries(sophus_linalg_proto_conv PUBLIC
  sophus_linalg
  sophus_linalg_proto_defs
  protobuf::libprotobuf
  farm_ng_core::farm_ng_core_proto_defs
  farm_ng_core::farm_ng_core_prototools
 )

if(${BUILD_SOPHUS_TESTS})
  farm_ng_add_test(conv
                PARENT_LIBRARY sophus_linalg_proto_conv
                LINK_LIBRARIES sophus_linalg_proto_conv
                LABELS small)
endif()
