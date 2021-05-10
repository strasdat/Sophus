#!/bin/bash

set -x # echo on
set -e # exit on error

git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 5.3.0
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j8
sudo make install
cd ../..
