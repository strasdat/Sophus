#[[
sophus_linalg



]]
farm_ng_add_library(sophus_linalg
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  rotation_matrix.h
)
target_link_libraries(sophus_linalg INTERFACE sophus_common)

if(BUILD_FARM_NG_PROTOS)
    add_subdirectory(proto)
endif()
