#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install \
  ccache \
  gfortran \
  libc++-dev \
  libatlas-base-dev \
  libgtest-dev \
  libgoogle-glog-dev \
  libprotobuf-dev \
  protobuf-compiler \
  libsuitesparse-dev \
  ninja-build

pip install git+https://github.com/gcovr/gcovr.git
