#!/bin/bash

set -x # echo on
set -e # exit on error

cd super_project
mkdir -p build
cd build
cmake ..
make -j2

make test
