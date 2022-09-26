farm_ng_add_protobufs(sophus_linalg_proto_defs
    NAMESPACE sophus
    PROTO_FILES
    sophus/linalg.proto)


farm_ng_add_protobufs(sophus_lie_proto_defs
      NAMESPACE sophus
      PROTO_FILES
      sophus/lie.proto
      DEPENDENCIES
      sophus_linalg_proto_defs)




farm_ng_add_protobufs(sophus_geometry_proto_defs
      NAMESPACE sophus
      PROTO_FILES
      sophus/geometry.proto
      DEPENDENCIES
      sophus_linalg_proto_defs)

farm_ng_add_protobufs(sophus_image_proto_defs
      NAMESPACE sophus
      PROTO_FILES
      sophus/image.proto
      DEPENDENCIES
      sophus_lie_proto_defs
      sophus_geometry_proto_defs)



farm_ng_add_protobufs(sophus_sensor_proto_defs
      NAMESPACE sophus
      PROTO_FILES
      sophus/sensor.proto
      DEPENDENCIES
      sophus_image_proto_defs)
