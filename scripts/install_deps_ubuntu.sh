#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install \
  ccache \
  gfortran \
  libc++-dev \
  libgtest-dev \
  libgoogle-glog-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
  libprotobuf-dev \
  protobuf-compiler

pip install git+https://github.com/gcovr/gcovr.git
