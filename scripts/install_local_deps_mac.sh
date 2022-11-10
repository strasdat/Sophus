#!/bin/bash

set -x # echo on
set -e # exit on error

brew install --verbose ceres-solver pre-commit ccache glog protobuf grpc
