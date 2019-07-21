#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install gfortran libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
tar xvf 3.3.4.tar.bz2
mkdir build-eigen
cd build-eigen
cmake ../eigen-eigen-5a0156e40feb -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT
sudo make install
cd ..

git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard afe93546b67cee0ad205fe8044325646ed5deea9
mkdir build
cd build
ccache -M 50G
ccache -s
cmake -DCXX11=On -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DOPENMP=Off ..
make -j3
sudo make install
cd ../..

git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 5.3.0
mkdir build
cd build
cmake ..
make -j3
sudo make install
cd ../..