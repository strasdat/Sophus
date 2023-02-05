#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

rm -rf farm-ng-cmake-build
rm -rf grpc-build
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

mkdir grpc-build
cd grpc-build
cmake -G Ninja -DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR} \
  -DgRPC_INSTALL:BOOL=ON \
  -DgRPC_BUILD_TESTS:BOOL=OFF \
  -DgRPC_BUILD_MSVC_MP_COUNT:STRING=-1 \
  -DgRPC_PROTOBUF_PROVIDER:STRING=package \
  -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG \
  -DgRPC_RE2_PROVIDER:STRING=package \
  -DgRPC_ZLIB_PROVIDER:STRING=package \
  ../grpc
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
