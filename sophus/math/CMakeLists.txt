#[[
sophus_math



]]

set(sophus_math_src_prefixes
  rotation_matrix
)

set(sophus_math_h)
foreach(x ${sophus_math_src_prefixes} )
    list(APPEND sophus_math_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_math INTERFACE
  ${sophus_math_h})
target_link_libraries(sophus_math INTERFACE sophus_core)
add_library (sophus::sophus_math ALIAS sophus_math)

target_sources(sophus_math INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_math_h})
