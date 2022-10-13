farm_ng_add_protobufs(sophus_linalg_proto_defs
    NAMESPACE Sophus
    PROTO_FILES
    sophus/linalg.proto)

farm_ng_add_protobufs(sophus_lie_proto_defs
      NAMESPACE Sophus
      PROTO_FILES
      sophus/lie.proto
      DEPENDENCIES
      Sophus::sophus_linalg_proto_defs)

farm_ng_add_protobufs(sophus_geometry_proto_defs
      NAMESPACE Sophus
      PROTO_FILES
      sophus/geometry.proto
      DEPENDENCIES
      Sophus::sophus_linalg_proto_defs)

farm_ng_add_protobufs(sophus_image_proto_defs
      NAMESPACE Sophus
      PROTO_FILES
      sophus/image.proto
      DEPENDENCIES
      Sophus::sophus_lie_proto_defs
      Sophus::sophus_geometry_proto_defs)

farm_ng_add_protobufs(sophus_sensor_proto_defs
      NAMESPACE Sophus
      PROTO_FILES
      sophus/sensor.proto
      DEPENDENCIES
      Sophus::sophus_image_proto_defs)
