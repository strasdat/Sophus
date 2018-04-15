#!/bin/bash

set -x # echo on
set -e # exit on error

wget https://cmake.org/files/v3.9/cmake-3.9.0-Linux-x86_64.sh
chmod +x cmake-3.9.0-Linux-x86_64.sh
set +x # echo off
sudo ./cmake-3.9.0-Linux-x86_64.sh  --skip-license --prefix=/usr/local
set -x # echo on
sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
cmake --version

sudo apt-get -qq update
sudo apt-get install gfortran libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev
sudo unlink /usr/bin/gcc && sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc
sudo unlink /usr/bin/g++ && sudo ln -s /usr/bin/g++-5 /usr/bin/g++
gcc --version
g++ --version
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
tar xvf 3.3.4.tar.bz2
mkdir build-eigen
cd build-eigen
cmake ../eigen-eigen-5a0156e40feb -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT
sudo make install
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
