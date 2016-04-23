## Sophus

C++ implementation of Lie Groups using Eigen. 

### Packaging

This is a maintained version of the original code developed by Hauke Strasdat.

The current release branch is *indigo*. Ros packages are available for indigo, jade & kinetic (built as a [3rdparty package](http://wiki.ros.org/bloom/Tutorials/ReleaseThirdParty)).

### Installation - CMake

```
$ cd Sophus
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Installation - ROS 3rd Party Package

Install in a catkin workspace dedicated to 3rd party packages (usually you'll install this package on its own):

```
$ mkdir -p ~/3rd_party_workspace/src
$ cd ~/3rd_party_workspace/src
$ wstool init .
$ wstool set sophus --git https://github.com/stonier/sophus.git --version=indigo
$ wstool update sophus
$ cd ~/3rd_party_workspace
$ catkin_make_isolated -DCMAKE_INSTALL_PREFIX=install_isolated --install
$ source ~/3rd_party_workspace/install_isolated/setup.bash
```

### Errata

Thanks to Steven Lovegrove, Sophus is now fully templated  - using the Curiously Recurring Template Pattern (CRTP).

(In order to go back to the non-templated/double-only version "git checkout a621ff".)


