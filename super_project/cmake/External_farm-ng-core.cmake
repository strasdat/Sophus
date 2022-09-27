ExternalProject_Add(farm-ng-core
    DEPENDS fmt expected eigen
    GIT_REPOSITORY  "https://github.com/farm-ng/farm-ng-core.git"
    GIT_TAG "main"
    PREFIX ${farm_ng_EXT_PREFIX}
    CMAKE_ARGS
    ${farm_ng_DEFAULT_ARGS}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DBUILD_FARM_NG_PROTOS=On
    )
