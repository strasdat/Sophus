#!/bin/bash

set -x # echo on
set -e # exit on error

brew install --verbose \
    ccache \
    glog \
    ninja
