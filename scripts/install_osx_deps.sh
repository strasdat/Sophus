#!/bin/bash

set -x # echo on
set -e # exit on error
brew update
brew install glog
brew install suite-sparse
git clone https://github.com/RLovelett/eigen.git eigen3
cd eigen3
git reset --hard 80d4cce2e3a06b137a93a5179eb8b0d6bc526ac0
mkdir build
cmake ..
cd ../..
wget http://ceres-solver.org/ceres-solver-1.12.0.tar.gz
tar zxf ceres-solver-1.12.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake -DCXX11=On ../ceres-solver-1.12.0
make -j3
make install
