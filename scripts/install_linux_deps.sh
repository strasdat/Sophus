#!/bin/bash

set -x # echo on
set -e # exit on error

sudo apt-get -qq update 
sudo apt-get install libeigen3-dev libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev
sudo unlink /usr/bin/gcc && sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc
sudo unlink /usr/bin/g++ && sudo ln -s /usr/bin/g++-5 /usr/bin/g++
gcc --version
g++ --version
wget http://ceres-solver.org/ceres-solver-1.12.0.tar.gz
tar zxf ceres-solver-1.12.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-1.12.0
make -j3
make test
sudo make install
