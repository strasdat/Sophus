#[[
sophus_calculus



]]

set(sophus_calculus_src_prefixes
  num_diff
)

set(sophus_calculus_h)
foreach(x ${sophus_calculus_src_prefixes} )
    list(APPEND sophus_calculus_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_calculus INTERFACE
  ${sophus_calculus_h})
target_link_libraries(sophus_calculus INTERFACE sophus_common)
add_library (sophus::sophus_calculus ALIAS sophus_calculus)

target_sources(sophus_calculus INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_calculus_h})
