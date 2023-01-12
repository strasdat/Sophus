#!/bin/bash

set -x # echo on
set -e # exit on error

brew install --verbose \
    ceres-solver \
    ccache \
    glog \
    protobuf \
    grpc
