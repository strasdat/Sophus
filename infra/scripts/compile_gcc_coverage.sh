#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir -p build
cd build
CC=clang CXX=clang++ cmake  -G Ninja -DSOPHUS_CERES=$SOPHUS_CERES -DROW_ACCESS=$ROW_ACCESS -DBUILD_FARM_NG_PROTOS=$BUILD_PROTOS -DCOVERAGE=On --debug-find ../../..
ninja

ninja test

#gcovr
