
set(sophus_geometry_proto_conv
  conv
)

set(sophus_geometry_proto_cpp)
set(sophus_geometry_proto_h)
foreach(x ${sophus_geometry_proto_conv} )
    list(APPEND sophus_geometry_proto_cpp ${x}.cpp)
    list(APPEND sophus_geometry_proto_h ${x}.h)
endforeach()

add_library(sophus_geometry_proto_conv SHARED
  ${sophus_geometry_proto_cpp}
  ${sophus_geometry_proto_h})
target_link_libraries(sophus_geometry_proto_conv PUBLIC
  sophus_lie protobuf::libprotobuf
  farm_ng_core::farm_ng_core_prototools
  sophus_geometry_proto_defs
  sophus_linalg_proto_conv)
add_library (farm_ng_core::sophus_geometry_proto_conv ALIAS sophus_geometry_proto_conv)
target_include_directories(sophus_geometry_proto_conv PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_sources(sophus_geometry_proto_conv INTERFACE
              FILE_SET HEADERS
              BASE_DIRS ../../..
              FILES ${sophus_geometry_proto_h})

if(${BUILD_SOPHUS_TESTS})
    farm_ng_add_test(conv
        PARENT_LIBRARY sophus_geometry_proto_conv
        LINK_LIBRARIES
        LABELS small)
endif()
