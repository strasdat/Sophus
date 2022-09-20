#!/bin/bash

set -x # echo on
set -e # exit on error

git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 8.1.1
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DBUILD_SHARED_LIBS=on -DFMT_TEST=off ..
make -j8
sudo make install
cd ../..
git clone https://github.com/TartanLlama/expected.git
cd expected
git checkout 96d547c03d2feab8db64c53c3744a9b4a7c8f2c5
mkdir -p build
cd build
cmake .. -DEXPECTED_BUILD_TESTS=off
sudo make install
cd ../..
