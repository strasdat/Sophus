#!/bin/bash
set -e

cd super_project
rm clang-tidy-build -rf
mkdir clang-tidy-build
cd clang-tidy-build
CC=clang CXX=clang++ cmake ..
make -j8
cd ../..

# run clang tidy
run-clang-tidy-10  -quiet -p super_project/clang-tidy-build/ cpp/sophus/.*\.cpp -fix -style none
