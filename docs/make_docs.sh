#!/bin/bash

set -x # echo on
set -e # exit on error

mkdir -p .doxyrest
cd .doxyrest/
pwd

if [[ ! -e doxyrest_b ]]; then
  git clone https://github.com/vovkos/doxyrest_b

  cd doxyrest_b
  git submodule update --init
  pwd

  cmake .
  cmake --build .
  cd ..
  pwd
fi
cd ..
pwd

doxygen doxyfile
.doxyrest/doxyrest_b/doxyrest/bin/Release/doxyrest -c doxyrest-config.lua

sphinx-build -b html -c . api_autogen html-dir
