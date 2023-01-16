# Create examples make targets using ctest
option(BUILD_SOPHUS_EXAMPLES "Build examples." ON)
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
add_library (Sophus::sophus ALIAS sophus)
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
