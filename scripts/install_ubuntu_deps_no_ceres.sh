#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt update -y
sudo apt install libc++-dev libgflags-dev libsuitesparse-dev clang

git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout c1d637433e3b3f9012b226c2c9125c494b470ae6

mkdir build-eigen
cd build-eigen
cmake .. -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT
sudo make install
cd ../..
