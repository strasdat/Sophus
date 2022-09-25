add_subdirectory(sophus)

add_library(sophus INTERFACE)
target_link_libraries(sophus INTERFACE
  sophus_common
  sophus_calculus
  sophus_linalg
  sophus_lie
  sophus_lie_interp
  sophus_geometry
)
add_library (sophus::sophus ALIAS sophus)

# Associate target with include directory
target_include_directories(sophus INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Create examples make targets using ctest
option(BUILD_SOPHUS_EXAMPLES "Build examples." ON)

set(sophus_install_targets
  sophus_common
  sophus_linalg
  sophus_calculus
  sophus_lie
  sophus_geometry
  sophus_lie_interp
  sophus_image
  sophus_sensor
  sophus_ceres
  sophus_experimental)

if(BUILD_FARM_NG_PROTOS)
  message(STATUS "Protobuf found")
  list(APPEND sophus_install_targets
    sophus_linalg_proto_conv sophus_linalg_proto_defs
    sophus_lie_proto_conv sophus_lie_proto_defs
    sophus_geometry_proto_conv sophus_geometry_proto_defs
    sophus_image_proto_conv sophus_image_proto_defs
    sophus_sensor_proto_conv sophus_sensor_proto_defs)
endif()


install(
  TARGETS
    ${sophus_install_targets}
  EXPORT SophusTargets
  FILE_SET HEADERS
  LIBRARY DESTINATION lib
)

write_basic_package_version_file(
    SophusConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(EXPORT SophusTargets
    FILE SophusTargets.cmake
    NAMESPACE sophus::
    DESTINATION share/sophus/cmake
)

configure_file(cmake/SophusConfig.cmake.in SophusConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/SophusConfigVersion.cmake"
        DESTINATION share/sophus/cmake)
