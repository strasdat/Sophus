#!/bin/bash
set -e

mkdir -p clang-tidy-build && cd clang-tidy-build && CC=clang CXX=clang++ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SOPHUS_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
cd ..

# run clang tidy
run-clang-tidy-10  -quiet -p clang-tidy-build/ sophus/.*\.cpp -fix
