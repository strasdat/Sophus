#!/bin/bash

set -x # echo on
set -e # exit on error

sudo apt-get -qq update
sudo apt-get install doxygen
sudo pip3 install --upgrade pip
sudo pip3 install exhale
