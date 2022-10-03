#[[
sophus_linalg



]]
farm_ng_add_library(sophus_linalg
  NAMESPACE sophus
  INCLUDE_DIR ../../
  HEADERS
  rotation_matrix.h
)
target_link_libraries(sophus_linalg INTERFACE sophus_common)

if(FARM_NG_BUILD_PROTOS)
    add_subdirectory(proto)
endif()
