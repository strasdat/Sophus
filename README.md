<h1 align="center"> Sophus </h1>

<p align="center">
  <i> A collection of c++ types for 2d and 3d geometric problems. </i>
</p>

<div align="center">

<a href="https://github.com/strasdat/Sophus/actions/workflows/build.yml">
  <img src="https://github.com/strasdat/Sophus/actions/workflows/build.yml/badge.svg"
       alt="Build CI Badge"/>
</a>
<a href="https://github.com/strasdat/Sophus/actions/workflows/format.yml">
  <img src="https://github.com/strasdat/Sophus/actions/workflows/format.yml/badge.svg"
       alt="Format CI Badge"/>
</a>
<a href="https://github.com/strasdat/Sophus/actions/workflows/docs.yml">
  <img src="https://github.com/strasdat/Sophus/actions/workflows/docs.yml/badge.svg"
       alt="Docs CI Badge"/>
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

<br>
<p align="center"><i> Manifolds, Image classes, Camera models and more. </i>
</p>
<br>

Sophus started as a c++ implementaion of Lie Groups / Manifolds. It evolved to a
collection of types and functions commonly used or 2d and 3d geometric problems
especially in the domain of `robotics`, `computer vision` annd `graphics`.

- [Lie groups / Manifold](#lie-groups) such as `SO(2)` and
  `SO(3)` to present rotations in 2d and 3,
- [Other geometric conecpts](#other-geometric-concepts) such as unit vector, plane
  fitting and inverse depth points,
- [Image classes](#image-classes)
- [Sensor models](#sensor-models) such `camera models` (as pinhole, orthographic
  and lens distortion model) and `IMU models`.
- [Sympy](#sympy) representations of selected types - for
  instance o symbolically derive Jacobians and autogenerate c++ code.
- [Serialization](#serialization) using proto (and soon json).

## 👟Getting Started

[Working in progress.]

Check out the docs for now: <https://strasdat.github.io/Sophus/latest/>

(Which are also work in progress...)

## 🌐Lie groups

### tldr: rotations, translations and scaling in 2d and 3d

`Lie groups` are generalizations of the Euclidean vector spaces R^N. A little
more formally, a Manifold which is also an [abstract group](https://en.wikipedia.org/wiki/Group_theory#Abstract_groups).

*Okay, and what is a Manifold?*

`Manifold` are generalizations of the Euclidean vector spaces R^N. In
particular, they behave locally like a Euclidean vector space, but globally
can have a very different structures. In particular there can be wrap-around.
The circle group SO(2) is the simplest example for such wrap around. Assume,
we have a dial pointing North. If you turn the dial 90 degree to the left, it
points West. If you turn it another 90 degrees it turns South. Now, if you turn
it again 90 degrees is points East. And you turn it left again for 90 degrees it
points North again. It wrapped around: `90 "+" 90 "+" 90 "+" 90 = 0`.

*Do I need to understand the concept of Manifold, Group Theory and Lie Groups
in order to use Sophus?*

Not at all! If you aim to solve geometric problems in 2d and 3d, it is best to
directly into concrete examples. By far the most commonly used Lie groups of
Sophus are the group of 3D rotations, also called Special Orthogonal Group,
short SO(3), as well as the group of rotation and translation in 3D, short
SE(3).

### 3d rotation example using the SO(3) type

```c++
  // The following demonstrates the group multiplication of rotation matrices

  // Create rotation matrices from rotations around the x and y and z axes:
  double const kPi = sophus::kPi<double>;
  sophus::Rotation3F64 R1 = sophus::Rotation3F64::fromRx(kPi / 4);
  sophus::Rotation3F64 R2 = sophus::Rotation3F64::fromRy(kPi / 6);
  sophus::Rotation3F64 R3 = sophus::Rotation3F64::fromRz(-kPi / 3);

  std::cout << "The rotation matrices are" << std::endl;
  std::cout << "R1:\n" << R1.matrix() << std::endl;
  std::cout << "R2:\n" << R2.matrix() << std::endl;
  std::cout << "R3:\n" << R3.matrix() << std::endl;
  std::cout << "Their product R1*R2*R3:\n"
            << (R1 * R2 * R3).matrix() << std::endl;
  std::cout << std::endl;

  // Rotation matrices can act on vectors
  Eigen::Vector3d x;
  x << 0.0, 0.0, 1.0;
  std::cout << "Rotation matrices can act on 3-vectors" << std::endl;
  std::cout << "x\n" << x << std::endl;
  std::cout << "R2*x\n" << R2 * x << std::endl;
  std::cout << "R1*(R2*x)\n" << R1 * (R2 * x) << std::endl;
  std::cout << "(R1*R2)*x\n" << (R1 * R2) * x << std::endl;
  std::cout << std::endl;

  // SO(3) are internally represented as unit quaternions.
  std::cout << "R1 in matrix form:\n" << R1.matrix() << std::endl;
  std::cout << "R1 in unit quaternion form:\n"
            << R1.unitQuaternion().coeffs() << std::endl;
  // Note that the order of coefficients of Eigen's quaternion class is
  // (imag0, imag1, imag2, real)
  std::cout << std::endl;
```

[hello_so3.cpp](cpp/examples/test_install_targets/hello_so3.cpp)

### 3d rotation + translation example using the SE(3) type

```c++
  // Example of create a rigid transformation from an SO(3) = 3D rotation and a
  // translation 3-vector:

  // Let use assume there is a camera in the world. First we describe its
  // orientation in the world reference frame.
  sophus::Rotation3F64 world_from_camera_rotation =
      sophus::Rotation3F64::fromRx(sophus::kPi<double> / 4);
  // Then the position of the camera in the world.
  Eigen::Vector3d camera_in_world(0.0, 0.0, 1.0);

  // The pose (position and orientation) of the camera in the world is
  // constructed by its orientation ``world_from_camera_rotation`` as well as
  // its position ``camera_in_world``.
  sophus::Isometry3F64 world_anchored_camera_pose(
      world_from_camera_rotation, camera_in_world);

  // SE(3) naturally representation is a 4x4 matrix which can be accessed using
  // the .matrix() method:
  std::cout << "world_anchored_camera_pose:\n"
            << world_anchored_camera_pose.matrix() << std::endl;
```

[hello_se3](cpp/examples/test_install_targets/hello_se3.cpp)

### Tabel of Lie Groups

The following table gives an overview of all Lie Groups in Sophus.

| c++ type                                      | Lie group name                                       | Description                                                                                                |
| ----------------------------------------------|------------------------------------------------------| ---------------------------------------------------------------------------------------------------------- |
| [`Rotation2<T>`](cpp/sophus/lie/so2.h)              | Special Orthogonal Group in 2D, SO(2)                | rotations in 2d, also called Circle Group, or just "angle"                                                 |
| [`Rotation3<T>`](cpp/sophus/lie/so3.h)              | Special Orthogonal Group in 3D, SO(3)                | rotations in 3d, 3D orientations                                                                           |
| [`Isometry2<T>`](cpp/sophus/lie/se2.h)              | Special Euclidean Group in 2D, SE(3)                 | rotations and translations in 2D, also called 2D rigid body transformations, 2d poses, plane isometries    |
| [`Isometry3<T>`](cpp/sophus/lie/se3.h)              | Special Euclidean Group in 3D, SE(3)                 | rotations and translations in 3D, also called rigid body transformations,6 DoF poses, Euclidean isometries |
| [`RxSo2<T>`](cpp/sophus/lie/rxso2.h)          | Direct product of SO(3) and scalar matrix, R x SO(2) | scaling and rotations in 2D                                                                                |
| [`RxSo3<T>`](cpp/sophus/lie/rxso3.h)          | Direct product of SO(3) and scalar matrix  R x SO(3) | scaling and rotations in 3D                                                                                |
| [`Similarity2<T>`](cpp/sophus/lie/sim2.h)            | Similarity Group in 2D, Sim(2)                       | scaling, rotations and translation in 2D                                                                   |
| [`Similarity3<T>`](cpp/sophus/lie/sim3.h)            | Similarity Group in 3D, Sim(3)                       | scaling, rotations and translation in 3D                                                                   |
| [`Cartesian2<T>`](cpp/sophus/lie/cartesian.h) | 2D Euclidean Vector Space, R^2                       | all vector spaces are trivial Lie groups, also called 2d translation group, the translation part of SE(2)  |
| [`Cartesian3<T>`](cpp/sophus/lie/cartesian.h) | 3D Euclidean Vector Space, R^3                       | all vector spaces are trivial Lie groups, also called 3d translation group, the translation part of SE(2)  |
| ----------------------------------------------|------------------------------------------------------| ---------------------------------------------------------------------------------------------------------- |

Supported advanced features on Lie groups:

- ✅ (linear) interpolation
- ✅ Spline interpolation
- ✅ Averaging (of more than two elements)

## 📐More Geometry

## 🌁Image classes

## 📷Sensor Models

## 📜Sympy

## 🫙Serialization

## 🧑🏽‍🏭Contribute

Contributions are welcome!

## 📝License

This project is licensed under the [MIT](https://opensource.org/licenses/MIT) license.

## 👩‍🚀Show your support

Give a ⭐️ if this project helped you!
