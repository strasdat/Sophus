#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir build
cd build
pwd
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_COMPILE_WARNING_AS_ERROR=On -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
# Ubuntu builds via Github actions run on 2-core virtual machines
make -j2
ctest --output-on-failure
