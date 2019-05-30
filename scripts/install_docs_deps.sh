#!/bin/bash

set -x # echo on
set -e # exit on error

sudo apt-get -qq update
sudo apt-get install doxygen liblua5.3-dev
pip3 install sphinx_rtd_theme
pip3 install sympy

git clone https://github.com/vovkos/doxyrest_b
cd doxyrest_b
git reset --hard 9af65a29669f59a2e0059a6cc2f43bc4d33278b6
git submodule update --init
mkdir build
cd build
cmake ..
cmake --build .

cd ../..