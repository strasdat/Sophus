#!/bin/bash

set -x # echo on
set -e # exit on error
brew update
brew install eigen
brew install glog
brew install suite-sparse
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard afe93546b67cee0ad205fe8044325646ed5deea9
mkdir build
cd build
cmake -DCXX11=On ..
make -j3
make install