#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

rm -rf ceres-solver-build

cd ../ci-submodules

mkdir ceres-solver-build
cd ceres-solver-build
cmake -G Ninja ../ceres-solver
ninja
sudo ninja install
cd ..

cd ../scripts
