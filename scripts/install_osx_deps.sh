#!/bin/bash

set -x # echo on
set -e # exit on error
brew update
brew install glog
brew install suite-sparse
git clone https://github.com/RLovelett/eigen.git eigen3
cd eigen3
git reset --hard abd0909838886c1c7a3c261c23a02950affee243
mkdir build
cd build
cmake ..
make install
cd ../..
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard afe93546b67cee0ad205fe8044325646ed5deea9
mkdir build
cd build
cmake -DCXX11=On ..
make -j3
make install