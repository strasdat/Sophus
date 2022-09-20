#!/bin/bash

set -x # echo on
set -e # exit on error

brew update
brew install eigen

git clone https://github.com/google/googletest
cd googletest
git checkout release-1.12.1
mkdir -p build
cd build
cmake ..
make
make install
