#!/bin/bash

set -x # echo on
set -e # exit on error
brew update
brew install glog
brew install suite-sparse
wget http://ceres-solver.org/ceres-solver-1.12.0.tar.gz
tar zxf ceres-solver-1.12.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake -DCXX11=On ../ceres-solver-1.12.0
make -j3
make test
make install
