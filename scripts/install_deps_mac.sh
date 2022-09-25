#!/bin/bash

set -x # echo on
set -e # exit on error

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvf eigen-3.4.0.tar.gz
mkdir -p build-eigen
cd build-eigen
cmake ../eigen-3.4.0 -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT
sudo make install
cd ..

wget https://github.com/macports/macports-base/releases/download/v2.7.1/MacPorts-2.7.1-11-BigSur.pkg
sudo installer -pkg ./MacPorts-2.7.1-11-BigSur.pkg -target /
export PATH=/opt/local/bin:/opt/local/sbin:$PATH

sudo port install protobuf3-cpp

git clone https://github.com/google/googletest
cd googletest
git checkout release-1.12.1
mkdir -p build
cd build
cmake ..
make
make install
