#[[
sophus_common



]]

set(sophus_common_src_prefixes
  common
  types
)

set(sophus_common_h)
foreach(x ${sophus_common_src_prefixes} )
    list(APPEND sophus_common_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_common INTERFACE
  ${sophus_common_h})
target_link_libraries(sophus_common INTERFACE Eigen3::Eigen farm_ng_core::farm_ng_core_logging)
add_library (sophus::core ALIAS sophus_common)

target_sources(sophus_common INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_common_h})
