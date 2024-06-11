<div align="center">

<a href="https://github.com/strasdat/Sophus/actions/workflows/main.yml">
  <img src="https://github.com/strasdat/Sophus/actions/workflows/main.yml/badge.svg"
       alt="Build CI Badge"/>
</a>
<a href="https://github.com/strasdat/Sophus/actions/workflows/pybind.yml">
<img src="https://github.com/strasdat/Sophus/actions/workflows/pybind.yml/badge.svg"
       alt="PyBind CI Badge"/>
</a>
<a href="https://github.com/strasdat/Sophus/actions/workflows/sympy.yml">
<img src="https://github.com/strasdat/Sophus/actions/workflows/sympy.yml/badge.svg"
       alt="Sympy CI Badge"/>
</a>
<a href="https://github.com/strasdat/Sophus/blob/sophus2/LICENSE.txt">
  <img src="https://img.shields.io/github/license/elangosundar/awesome-README-templates?color=2b9348"
       alt="License Badge"/>
</a>

</div>

<h1 align="center"> Sophus </h1>

<p align="center">
  <i> 2d and 3d Lie Groups for Computer Vision and Robotics </i>
</p>

This is a c++ implementation of Lie groups commonly used for 2d and 3d
geometric problems (i.e. for Computer Vision or Robotics applications).
Among others, this package includes the special orthogonal groups SO(2) and
SO(3) to present rotations in 2d and 3d as well as the special Euclidean group
SE(2) and SE(3) to represent isometries also known as rigid body transformations
(i.e. rotations and translations) in 2d and 3d.

## Status


*Sophus (aka Sophus 1) is in maintenance mode. As of June 2024, there is no
plane to add new larger features and future PRs will likely be limited to bug
fixes, small improvements and toolchain updates.*

However, next incarnations of Sophus are under development:


 - sophus2 is the next c++ iteration of Sophus and is a complete rewrite.
   In addition to the Lie groups, it includes a more geometric concepts
   such unit vector, splines, image classes, camera models and more.

   It is currently hosted as part of the [farm-ng-core repository](https://github.com/farm-ng/farm-ng-core/tree/cygnet-dev)
   and has likely only a few community users. While the code itself is in a good shape, there are
   no good build instructions yet. Hopefully, this will change in the near future.


 - sophus-rs is a Rust version of Sophus. Similar to sophus2, it includes a more geometric concepts
   such unit vector, splines, image classes, camera models and more. Also it includes an early and
   experimental version of non-linear least squares optimization library (similar to Ceres, g2o,
   etc.).

   sophus-rs has likely only a few community users so far, but should be easy to build and
   experiment with - of course being written in Rust.

   https://github.com/sophus-vision/sophus-rs

   https://crates.io/crates/sophus



How to build Sophus from source
-------------------------------

Sophus requires a C++17 compiler (though older versions build with C++14).

Sophus is tested on Linux and macOS. It also worked on Windows in the past, however there is
currently no CI for Windows, so it might require some smaller patches to build on Windows.

There are no comprehensive build instructions but inspecting the install [scripts](scripts/)
as well as the [main.yml](.github/workflows/main.yml) file should give you a good idea how to
build the required dependencies.

Installing Sophus through vcpkg
-------------------------------

You can build and install Sophus using [vcpkg](https://github.com/Microsoft/vcpkg/) dependency manager::

```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install sophus
```

The Sophus port in vcpkg is kept up to date by Microsoft team members and community contributors.
If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg)
on the vcpkg repository.
