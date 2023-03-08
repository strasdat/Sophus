---
sidebar_position: 2
---

# 🌐Lie Groups

# Lie Gropus

| c++ type        | Lie group name                                       | Description                                                                                                | minimal representation      | #DoF | Matrix representation | compact internal manifold representation    | #params |
| ----------------|------------------------------------------------------| ---------------------------------------------------------------------------------------------------------- | ----------------------------| -----|---------------------- | ------------------------------------------- | ------- |
| `Rotation2<T>`        | Special Orthogonal Group in 2D, SO(2)                | rotations in 2d, also called Circle Group, or just "angle"                                                 | rotation angle              | 1    | 2x2 matrix            | unit complex number                         | 2       |
| `Rotation3<T>`        | Special Orthogonal Group in 3D, SO(3)                | rotations in 3d, 3D orientations                                                                           | rotation vector             | 3    | 3x3 matrix            | unit quaternion number                      | 4       |
| `Isometry2<T>`        | Special Euclidean Group in 2D, SE(3)                 | rotations and translations in 2D, also called 2D rigid body transformations, 2d poses, plane isometries    | tangent vector of SE(2)     | 3    | 3x3 matrix            | unit complex number + translation vector    | 2+2 = 4 |
| `Isometry3<T>`        | Special Euclidean Group in 3D, SE(3)                 | rotations and translations in 3D, also called rigid body transformations,6 DoF poses, Euclidean isometries | tangent vector of R x SE(3) | 6    | 4x4 matrix            | unit quaternion number + translation vector | 4+3 = 7 |
| `RxSo2<T>`      | Direct product of SO(3) and scalar matrix, R x SO(2) | scaling and rotations in 2D                                                                                | tangent vector of R x SO(2) | 3    | 2x2 matrix            | non-zero complex number                     | 2       |
| `RxSo3<T>`      | Direct product of SO(3) and scalar matrix  R x SO(3) | scaling and rotations in 3D                                                                                | tangent vector of R x SO(3) | 4    | 3x3 matrix            | non-zero quaternion number                  | 4       |
| `Similarity2<T>`       | Similarity Group in 2D, Sim(2)                       | scaling, rotations and translation in 2D                                                                   | tangent vector of Sim(2)    | 4    | 3x3 matrix            | non-zero complex number+ translation vector | 2+2 = 4 |
| `Similarity3<T>`       | Similarity Group in 3D, Sim(3)                       | scaling, rotations and translation in 3D                                                                   | tangent vector of Sim(3)    | 4    | 4x4 matrix            | non-zero complex number+ translation vector | 4+3 = 7 |
| `Cartesian2<T>` | 2D Euclidean Vector Space, R^2                       | all vector spaces are trivial Lie groups, also called 2d translation group, the translation part of SE(2)  | 2-vector                    | 2    | 3x3 matrix            | 2-vector                                    | 2       |
| `Cartesian3<T>` | 3D Euclidean Vector Space, R^3                       | all vector spaces are trivial Lie groups, also called 3d translation group, the translation part of SE(2)  | 3-vector                    | 3    | 4x4 matrix            | 3-vector                                    | 3       |
