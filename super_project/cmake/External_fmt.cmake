ExternalProject_Add(fmt
    GIT_REPOSITORY  "https://github.com/fmtlib/fmt.git"
    GIT_TAG "8.1.1"
    PREFIX ${farm_ng_EXT_PREFIX}
    CMAKE_ARGS
    ${farm_ng_DEFAULT_ARGS}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DFMT_TEST:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=ON
    )
