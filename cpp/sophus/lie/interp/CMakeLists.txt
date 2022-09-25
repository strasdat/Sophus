#[[
sophus_lie_interp



]]

set(sophus_lie_interp_src_prefixes
  average
  interpolate
  spline
  details/interpolate_impl
)

set(sophus_lie_interp_h)
foreach(x ${sophus_lie_interp_src_prefixes} )
    list(APPEND sophus_lie_interp_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_lie_interp INTERFACE
  ${sophus_lie_interp_h}
)
add_library (sophus::sophus_lie_interp ALIAS sophus_lie_interp)
target_link_libraries(sophus_lie_interp INTERFACE sophus_lie)

target_sources(sophus_lie_interp INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_lie_interp_h})
