#!/bin/bash

set -x # echo on
set -e # exit on error

cd super_project
mkdir -p build
cd build
cmake ..
make -j2

# The make runs the tests in each of the projects built
# there is no seperate test target
