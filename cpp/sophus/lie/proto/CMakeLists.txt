farm_ng_add_library(sophus_lie_proto_conv
  NAMESPACE Sophus
  INCLUDE_DIR ../../..
  HEADERS
    conv.h
  SOURCES
    conv.cpp
)

target_link_libraries(sophus_lie_proto_conv PUBLIC
  sophus_lie
  protobuf::libprotobuf
  sophus_linalg_proto_conv
  farm_ng_core::farm_ng_core_prototools
  sophus_lie_proto_defs
  )


if(${BUILD_SOPHUS_TESTS})
  farm_ng_add_test(conv
                PARENT_LIBRARY sophus_lie_proto_conv
                LINK_LIBRARIES sophus_lie_proto_conv
                LABELS small)
endif()
