#[[
sophus_linalg



]]

set(sophus_linalg_src_prefixes
  rotation_matrix
)

set(sophus_linalg_h)
foreach(x ${sophus_linalg_src_prefixes} )
    list(APPEND sophus_linalg_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()

add_library(sophus_linalg INTERFACE
  ${sophus_linalg_h})
target_link_libraries(sophus_linalg INTERFACE sophus_common)
add_library (sophus::sophus_linalg ALIAS sophus_linalg)

target_sources(sophus_linalg INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_linalg_h})

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()
