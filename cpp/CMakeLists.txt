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

install(TARGETS sophus
    EXPORT SophusTargets
    LIBRARY DESTINATION lib
    COMPONENT Libs
  )

install(EXPORT SophusTargets
    FILE SophusTargets.cmake
    NAMESPACE sophus::
    DESTINATION share/sophus/cmake
)

# Create examples make targets using ctest
option(BUILD_SOPHUS_EXAMPLES "Build examples." ON)


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

# This allows others to find Sophus without installing it
# https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html#exporting-targets-from-the-build-tree
export(EXPORT SophusTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/SophusTargets.cmake"
    NAMESPACE sophus::
)

configure_file(cmake/SophusConfig.cmake.in SophusConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SophusConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/SophusConfigVersion.cmake"
        DESTINATION share/sophus/cmake)
