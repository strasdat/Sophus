#[[
sophus_ceres



]]

set(sophus_ceres_src_prefixes
  jet_helpers
  manifold
  typetraits
)

set(sophus_ceres_h)
foreach(x ${sophus_ceres_src_prefixes} )
    list(APPEND sophus_ceres_h ${CMAKE_CURRENT_SOURCE_DIR}/${x}.h)
endforeach()


add_library(sophus_ceres INTERFACE
  ${sophus_ceres_h}
)
target_link_libraries(sophus_ceres INTERFACE sophus_lie)
add_library (sophus::sophus_ceres ALIAS sophus_ceres)

target_sources(sophus_ceres INTERFACE
               FILE_SET HEADERS
               BASE_DIRS ../..
               FILES ${sophus_ceres_h})


set(sophus_ceres_src_prefixes
  rxso2
  rxso3
  se2
  se3
  sim2
  sim3
  so2
  so3
)

if(${BUILD_SOPHUS_TESTS})
  find_package(Ceres 2.1.0 REQUIRED)
  foreach(test_basename ${sophus_ceres_src_prefixes})
  farm_ng_add_test(${test_basename}
                        PARENT_LIBRARY sophus_ceres
                        LINK_LIBRARIES
                        Ceres::ceres
                        LABELS large)
  target_compile_definitions(${test_basename}_test_sophus_ceres PRIVATE ${ARGN})
  endforeach()
endif()
