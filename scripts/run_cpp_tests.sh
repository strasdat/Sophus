#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_SOPHUS_TESTS=ON -DBUILD_FARM_NG_PROTOS=$BUILD_PROTOS ..
make -j2
make CTEST_OUTPUT_ON_FAILURE=1 test
