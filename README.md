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
<a href="https://github.com/strasdat/Sophus//blob/23.04-beta/LICENSE">
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
- fitting and inverse depth points,
- [Image classes](#image-classes)
- [Sensor models](#sensor-models) such `camera models` (as pinhole, orthographic
  and lens distortion model) and `IMU models`.
- [Sympy](#sympy) representations of selected types - for
  instance o symbolically derive Jacobians and autogenerate c++ code.
- [Serialization](#serialization) using proto (and soon json).

## Getting Started

[Working in progress.]

Check out the docs for now: https://strasdat.github.io/Sophus/latest/

(Which are also work in progress...)

## Lie groups

### tldr: rotations, translations and scaling in 2d and 3d

`Lie groups` are generalizations of the Euclidean vector spaces R^N. A little
more formally, a Manifold which is also an [abstract group](https://en.wikipedia.org/wiki/Group_theory#Abstract_groups).


### Okay, and what is a Manifold?

`Manifold` are generalizations of the Euclidean vector spaces R^N. In
particular, they behave locally like a Euclidean vector space, but globally
can have a very different structures. In particular there can be wrap-around.
The circle group SO(2) is the simplest example for such wrap around. Assume,
we have a dial pointing North. If you turn the dial 90 degree to the left, it
points West. If you turn it another 90 degrees it turns South. Now, if you turn
it again 90 degrees is points East. And you turn it left again for 90 degrees it
points North again. It wrapped around: `90 "+" 90 "+" 90 "+" 90 = 0`.

### Do I need to understand the concept of Manifold, Group Theory and Lie Groups in order to use Sophus?

Not at all! If you aim to solve geometric problems in 2d and 3d, I'd suggest to
study the table below, in particular the description column, to see what type
of transformations is supported bie the ``lie`` component of Sophus. The most
commonly used Lie group of Sophus is SE(3). Possibly you might want to start
studying this one. If you are interested in 2d problems, such as 2d mapping,
SE(2) might be a good entry point.


| c++ type        | Lie group                                            | Description                                                                                                | minimal representation      | #DoF | Matrix representation | compact internal manifold representation    | #params |
| ----------------|------------------------------------------------------| ---------------------------------------------------------------------------------------------------------- | ----------------------------| -----|---------------------- | ------------------------------------------- | ------- |
| `So2<T>`        | Special Orthogonal Group in 2D, SO(2)                | rotations in 2d, also called Circle Group, or just "angle"                                                 | rotation angle              | 1    | 2x2 matrix            | unit complex number                         | 2       |
| `So3<T>`        | Special Orthogonal Group in 3D, SO(3)                | rotations in 3d, 3D orientations                                                                           | rotation vector             | 3    | 3x3 matrix            | unit quaternion number                      | 4       |
| `Se2<T>`        | Special Euclidean Group in 2D, SE(3)                 | rotations and translations in 2D, also called 2D rigid body transformations, 2d poses, plane isometries    | tangent vector of SE(2)     | 3    | 3x3 matrix            | unit complex number + translation vector    | 2+2 = 4 |
| `Se3<T>`        | Special Euclidean Group in 3D, SE(3)                 | rotations and translations in 3D, also called rigid body transformations,6 DoF poses, Euclidean isometries | tangent vector of R x SE(3) | 6    | 4x4 matrix            | unit quaternion number + translation vector | 4+3 = 7 |
| `RxSo2<T>`      | Direct product of SO(3) and scalar matrix, R x SO(2) | scaling and rotations in 2D                                                                                | tangent vector of R x SO(2) | 3    | 2x2 matrix            | non-zero complex number                     | 2       |
| `RxSo3<T>`      | Direct product of SO(3) and scalar matrix  R x SO(3) | scaling and rotations in 3D                                                                                | tangent vector of R x SO(3) | 4    | 3x3 matrix            | non-zero quaternion number                  | 4       |
| `Sim2<T>`       | Similarity Group in 2D, Sim(2)                       | scaling, rotations and translation in 2D                                                                   | tangent vector of Sim(2)    | 4    | 3x3 matrix            | non-zero complex number+ translation vector | 2+2 = 4 |
| `Sim3<T>`       | Similarity Group in 3D, Sim(3)                       | scaling, rotations and translation in 3D                                                                   | tangent vector of Sim(3)    | 4    | 4x4 matrix            | non-zero complex number+ translation vector | 4+3 = 7 |
| `Cartesian2<T>` | 2D Euclidean Vector Space, R^2                       | all vector spaces are trivial Lie groups, also called 2d translation group, the translation part of SE(2)  | 2-vector                    | 2    | 3x3 matrix            | 2-vector                                    | 2       |
| `Cartesian3<T>` | 3D Euclidean Vector Space, R^3                       | all vector spaces are trivial Lie groups, also called 3d translation group, the translation part of SE(2)  | 3-vector                    | 3    | 4x4 matrix            | 3-vector                                    | 3       |
| ----------------|------------------------------------------------------| ---------------------------------------------------------------------------------------------------------- | ----------------------------| -----|---------------------- | ------------------------------------------- | ------- |

Supported advanced features on Lie groups:

[x] (linear) interpolation
[x] Spline interpolation
[x] Averaging (of more than two elements)

## Other geometric concepts

## Image classes

## Sensor Models

## Sympy

## Serialization

## Contribute

Contributions are welcome!

## :pencil: License

This project is licensed under [MIT](https://opensource.org/licenses/MIT) license.

## :man_astronaut: Show your support

Give a ⭐️ if this project helped you!
