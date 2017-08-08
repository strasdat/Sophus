#!/bin/bash

set -x # echo on
set -e # exit on error

sudo apt-get -qq update 
sudo apt-get install libeigen3-dev libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev
sudo unlink /usr/bin/gcc && sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc
sudo unlink /usr/bin/g++ && sudo ln -s /usr/bin/g++-5 /usr/bin/g++
gcc --version
g++ --version
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard afe93546b67cee0ad205fe8044325646ed5deea9
mkdir build
cd build
cmake -DCXX11=On ..
make -j3
sudo make install