#!/bin/bash

set -x # echo on
set -e # exit on error

sudo port install cmake protobuf3-cpp ccache pre-commit google-glog
