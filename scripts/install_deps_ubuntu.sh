#!/bin/bash

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install gfortran libc++-dev libgtest-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libprotobuf-dev protobuf-compiler ccache

