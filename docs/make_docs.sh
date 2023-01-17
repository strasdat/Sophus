#!/bin/bash

set -x # echo on
set -e # exit on error

cd c++-api
if [[ ! -e doxygen-awesome-css ]]; then
  git submodule add https://github.com/jothepro/doxygen-awesome-css.git
  cd doxygen-awesome-css
  git checkout v2.1.0
  cd ..
fi
pwd

doxygen doxyfile
cd ..
sphinx-build -b html . html-dir
