#!/bin/bash

set -x # echo on
set -e # exit on error

wget https://github.com/macports/macports-base/releases/download/v2.7.1/MacPorts-2.7.1-11-BigSur.pkg
sudo installer -pkg ./MacPorts-2.7.1-11-BigSur.pkg -target /
export PATH=/opt/local/bin:/opt/local/sbin:$PATH

sudo PATH=$PATH port install protobuf3-cpp
