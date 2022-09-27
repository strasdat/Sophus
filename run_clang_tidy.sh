#!/bin/bash
set -e

cd super_project
rm clang-tidy-build -rf
mkdir clang-tidy-build
cd clang-tidy-build
CC=clang CXX=clang++ cmake -DSUPER_PROJ_FARM_NG_PROTOS=OFF ..
make -j2
cd ../..

# run clang tidy
run-clang-tidy-10  -quiet -p super_project/clang-tidy-build/Sophus-build cpp/sophus/.*\.cpp -fix -style none
