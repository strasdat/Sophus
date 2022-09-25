#!/bin/bash

set -x # echo on
set -e # exit on error

export PATH=/opt/local/bin:/opt/local/sbin:$PATH
sudo port install gflags google-glog SuiteSparse

git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard b0aef211db734379319c19c030e734d6e23436b0
ls
rm -rf build
mkdir build
cd build
ccache -s
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DBUILD_TESTING=OFF ..
make -j8
sudo make install
cd ../..
