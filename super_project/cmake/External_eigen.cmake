ExternalProject_Add(eigen
    GIT_REPOSITORY https://github.com/hexagon-geo-surv/eigen.git
    GIT_TAG "3.4.0"
    PREFIX ${farm_ng_EXT_PREFIX}
    CMAKE_ARGS
    ${farm_ng_DEFAULT_ARGS}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DEIGEN_DEFAULT_TO_ROW_MAJOR=$EIGEN_DEFAULT_TO_ROW_MAJOR
)
