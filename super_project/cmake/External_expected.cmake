ExternalProject_Add(expected
    GIT_REPOSITORY  "https://github.com/TartanLlama/expected.git"
    GIT_TAG "96d547c03d2feab8db64c53c3744a9b4a7c8f2c5"
    PREFIX ${farm_ng_EXT_PREFIX}
    CMAKE_ARGS
    ${farm_ng_DEFAULT_ARGS}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DEXPECTED_BUILD_TESTS=off
)
