#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

rm -rf ceres-build

cd ../ci-submodules

mkdir ceres-build
cd ceres-build
cmake -G Ninja ../ceres
ninja
sudo ninja install
cd ..

cd ../scripts
