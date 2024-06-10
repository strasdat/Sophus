#!/bin/bash

set -x # echo on
set -e # exit on error

brew update
brew install ccache

# Get dependencies for Ceres Solver
brew install eigen
brew install gflags
brew install glog
brew install gcc
brew install openblas
brew install libomp
brew install hwloc
brew install tbb

git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard 6a74af202d83cf31811ea17dc66c74d03b89d79e
mkdir target
cd target
ls
ccache -s
cmake -DMINIGLOG=On -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j8
sudo make install
cd ../..
