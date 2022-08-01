#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install gfortran libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libceres-dev ccache
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.tar.bz2
tar xvf eigen-3.3.4.tar.bz2
mkdir build-eigen
cd build-eigen
cmake ../eigen-3.3.4 -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT
sudo make install
cd ..

git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard b0aef211db734379319c19c030e734d6e23436b0
mkdir build
cd build
ccache -s
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j8
sudo make install
cd ../..
