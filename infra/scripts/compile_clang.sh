#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir -p build
cd build
CC=clang CXX=clang++ cmake -G Ninja -DROW_ACCESS=$ROW_ACCESS -DBUILD_FARM_NG_PROTOS=$BUILD_PROTOS -DBUILD_SOPHUS_TESTS=On --debug-find ../../..
ninja
