#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir -p build
cd build
cmake -DSOPHUS_CERES=$SOPHUS_CERES -DROW_ACCESS=$ROW_ACCESS -DBUILD_FARM_NG_PROTOS=$BUILD_PROTOS -DBUILD_SOPHUS_TESTS=On --debug-find ../../..
make -j1
