#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install gfortran libc++-dev libgtest-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev ccache
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvf eigen-3.4.0.tar.gz
mkdir -p build-eigen
cd build-eigen
cmake ../eigen-3.4.0 -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT
sudo make install
cd ..
