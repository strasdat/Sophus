#!/bin/bash

set -x # echo on
set -e # exit on error
brew update
brew install eigen
brew install glog
brew install suite-sparse
brew install ccache
export PATH="/usr/local/opt/ccache/libexec:$PATH"
whereis ccache
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard 399cda773035d99eaf1f4a129a666b3c4df9d1b1
mkdir build
cd build
ccache -M 50G
ccache -s
cmake -DCXX11=On -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j3
make install
