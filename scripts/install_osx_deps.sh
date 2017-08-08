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
cmake ..
make install
cd ../..
wget http://ceres-solver.org/ceres-solver-1.12.0.tar.gz
tar zxf ceres-solver-1.12.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake -DCXX11=On ../ceres-solver-1.12.0
make -j3
make install
