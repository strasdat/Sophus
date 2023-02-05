#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

rm -rf farm-ng-cmake-build
rm -rf farm-ng-core-build

cd ../ci-submodules

mkdir farm-ng-cmake-build
cd farm-ng-cmake-build
cmake -G Ninja ../farm-ng-cmake
ninja
sudo ninja install
cd ..

mkdir farm-ng-core-build
cd farm-ng-core-build
cmake -G Ninja -DBUILD_FARM_NG_PROTOS=On ../farm-ng-core
ninja
sudo ninja install
cd ..

cd ../scripts
