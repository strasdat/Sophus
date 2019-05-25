#!/bin/bash

set -x # echo on
set -e # exit on error

sudo apt-get -qq update
sudo apt-get install doxygen
pip3 install exhale
