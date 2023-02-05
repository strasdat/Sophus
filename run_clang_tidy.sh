#!/bin/bash
set -e

rm -rf clang-tidy-build
mkdir clang-tidy-build
cd clang-tidy-build
CC=clang CXX=clang++ cmake -GNinja -DSOPHUS_CERES=Off -DBUILD_FARM_NG_PROTOS=Off ..
ninja
cd ..

# run clang tidy
run-clang-tidy-12 -quiet -p clang-tidy-build/ cpp/sophus/.*\.cpp -fix -style none
