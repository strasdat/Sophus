#!/bin/bash
set -e

rm clang-tidy-build -rf
mkdir clang-tidy-build
cd clang-tidy-build
CC=clang CXX=clang++ cmake -GNinja -DBUILD_FARM_NG_PROTOS=$BUILD_PROTOS ..
ninja
cd ..

# run clang tidy
run-clang-tidy-12 -quiet -p super_project/clang-tidy-build/Sophus-build cpp/sophus/.*\.cpp -fix -style none
