add_subdirectory(sophus)

add_library(sophus INTERFACE)
target_link_libraries(sophus INTERFACE
  sophus_core
  sophus_math
  sophus_lie
  sophus_geometry
  sophus_interp
)
add_library (sophus::sophus ALIAS sophus)

# Associate target with include directory
target_include_directories(sophus INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Create examples make targets using ctest
option(BUILD_SOPHUS_EXAMPLES "Build examples." ON)

install(
  TARGETS
     sophus_core
     sophus_math
     sophus_lie
     sophus_geometry
     sophus_interp
     sophus_image
     sophus_sensor
     sophus_ceres
     sophus_experimental
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
