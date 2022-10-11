#!/bin/bash

set -x # echo on
set -e # exit on error

sudo PATH=$PATH port install protobuf3-cpp
sudo port install google-glog
