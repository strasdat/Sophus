#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

rm -rf farm-ng-cmake-build
rm -rf fmt-build
rm -rf expected-build
rm -rf farm-ng-core-build
rm -rf eigen-build
rm -rf Sophus-build

cd ../ci-submodules

mkdir farm-ng-cmake-build
cd farm-ng-cmake-build
cmake -G Ninja ../farm-ng-cmake
ninja
sudo ninja install
cd ..

mkdir fmt-build
cd fmt-build
cmake -G Ninja -DBUILD_SHARED_LIBS=On -DFMT_TEST=Off ../fmt
ninja
sudo ninja install
cd ..

mkdir expected-build
cd expected-build
cmake -G Ninja -DEXPECTED_BUILD_TESTS=Off ../expected
ninja
sudo ninja install
cd ..

mkdir farm-ng-core-build
cd farm-ng-core-build
cmake -G Ninja -DBUILD_FARM_NG_PROTOS=Off ../farm-ng-core
ninja
sudo ninja install
cd ..

mkdir eigen-build
cd eigen-build
cmake -G Ninja ../eigen
ninja
sudo ninja install
cd ..

cd ../scripts
