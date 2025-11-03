#!/bin/bash

set -x # echo on
set -e # exit on error

brew update
brew install ccache

# Get dependencies for Ceres Solver
brew install abseil
brew install eigen
brew install gflags
brew install glog
brew install gcc
brew install googletest
brew install openblas
brew install libomp
brew install hwloc
brew install tbb

git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard f9b7b6651b108136a16df44d91fb31735645f5a7
mkdir target
cd target
ls
ccache -s
cmake -DMINIGLOG=On -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j8
sudo make install
cd ../..
