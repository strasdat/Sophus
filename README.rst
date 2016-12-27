Sophus
======

Overview
--------

This is a c++-11 implementation of Lie groups commonly used for 2d and 3d
geometric problems (i.e. for Computer Vision or Robotics applications).
Among others, this package includes the special orthogonal groups SO(2) and
SO(3) to present rotations in 2d and 3d as well as the special Euclidean group
SE(2) and SE(3) to represent rigid body transformations (i.e. rotations and
translations) in 2d and 3d.

Cross platform support
----------------------

Sophus supports clang and gcc on Linux and OS X as well as msvc on Windows.


The specific compiler and operating system versions which are supported are the
ones which are used in the Continuous Integration (CI):

linux, os x:

.. image:: https://travis-ci.org/strasdat/Sophus.svg?branch=master

windows:

.. image:: https://travis-ci.org/strasdat/Sophus.svg?branch=master

Code coverage:

.. image:: https://coveralls.io/repos/github/strasdat/Sophus/badge.svg?branch=master


However, it should work (with no to minor modification) on many other
modern configurations which support CMake and c++11 as well.