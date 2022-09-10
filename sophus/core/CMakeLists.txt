#[[
sophus_core



]]

set(sophus_core_src_prefixes
  common
  types
)

set(sophus_core_h)
foreach(x ${sophus_core_src_prefixes} )
    list(APPEND sophus_core_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_core INTERFACE
  ${sophus_core_h})
target_link_libraries(sophus_core INTERFACE Eigen3::Eigen farm_ng_core_logging)
add_library (sophus::core ALIAS sophus_core)

target_sources(sophus_core INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_core_h})
