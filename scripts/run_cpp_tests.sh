#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir build
cd build
pwd
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DUSE_BASIC_LOGGING=$USE_BASIC_LOGGING -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
# Ubuntu builds via Github actions run on 2-core virtual machines
make -j2
make CTEST_OUTPUT_ON_FAILURE=1 test
