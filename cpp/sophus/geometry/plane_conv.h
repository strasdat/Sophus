// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Transformations between poses and hyperplanes.

#pragma once

#include "sophus/common/common.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"

namespace sophus {

/// Takes in a rotation ``foo_rotation_plane`` and returns the corresponding
/// line normal along the y-axis (in reference frame ``foo``).
///
template <class TScalar>
auto normalFromRotation2(Rotation2<TScalar> const& foo_rotation_line)
    -> Eigen::Vector2<TScalar> {
  return foo_rotation_line.matrix().col(1);
}

/// Takes in line normal in reference frame foo and constructs a corresponding
/// rotation matrix ``foo_rotation_line``.
///
/// Precondition: ``normal_in_foo`` must not be close to zero.
///
template <class TScalar>
auto rotation2FromNormal(Eigen::Vector2<TScalar> normal_in_foo)
    -> Rotation2<TScalar> {
  SOPHUS_ASSERT(
      normal_in_foo.squaredNorm() > kEpsilon<TScalar>,
      "{}",
      normal_in_foo.transpose().eval());
  normal_in_foo.normalize();
  return Rotation2<TScalar>::fromParams(
      {normal_in_foo.y(), -normal_in_foo.x()});
}

/// Takes in a rotation ``foo_rotation_plane`` and returns the corresponding
/// plane normal along the z-axis (in reference frame ``foo``).
///
template <class TScalar>
auto normalFromRotation3(Rotation3<TScalar> const& foo_rotation_plane)
    -> Eigen::Vector3<TScalar> {
  return foo_rotation_plane.matrix().col(2);
}

/// Takes in plane normal in reference frame foo and constructs a corresponding
/// rotation matrix ``foo_rotation_plane``.
///
/// Note: The ``plane`` frame is defined as such that the normal points along
///       the positive z-axis. One can specify hints for the x-axis and y-axis
///       of the ``plane`` frame.
///
/// Preconditions:
/// - ``normal_in_foo``, ``xDirHint_foo``, ``yDirHint_foo`` must not be close to
///   zero.
/// - ``xDirHint_foo`` and ``yDirHint_foo`` must be approx. perpendicular.
///
template <class TScalar>
auto rotation3FromNormal(
    Eigen::Vector3<TScalar> const& normal_in_foo,
    Eigen::Vector3<TScalar> x_dir_hint_foo =
        Eigen::Vector3<TScalar>(TScalar(1), TScalar(0), TScalar(0)),
    Eigen::Vector3<TScalar> y_dir_hint_foo = Eigen::Vector3<TScalar>(
        TScalar(0), TScalar(1), TScalar(0))) -> Eigen::Matrix3<TScalar> {
  SOPHUS_ASSERT(
      x_dir_hint_foo.dot(y_dir_hint_foo) < kEpsilon<TScalar>,
      "xDirHint ({}) and yDirHint ({}) must be perpendicular.",
      x_dir_hint_foo.transpose(),
      y_dir_hint_foo.transpose());
  using std::abs;
  using std::sqrt;
  TScalar const x_dir_hint_foo_sqr_length = x_dir_hint_foo.squaredNorm();
  TScalar const y_dir_hint_foo_sqr_length = y_dir_hint_foo.squaredNorm();
  TScalar const normal_foo_sqr_length = normal_in_foo.squaredNorm();
  SOPHUS_ASSERT(
      x_dir_hint_foo_sqr_length > kEpsilon<TScalar>,
      "{}",
      x_dir_hint_foo.transpose());
  SOPHUS_ASSERT(
      y_dir_hint_foo_sqr_length > kEpsilon<TScalar>,
      "{}",
      y_dir_hint_foo.transpose());
  SOPHUS_ASSERT(
      normal_foo_sqr_length > kEpsilon<TScalar>,
      "{}",
      normal_in_foo.transpose());

  Eigen::Matrix3<TScalar> basis_foo;
  basis_foo.col(2) = normal_in_foo;

  if (abs(x_dir_hint_foo_sqr_length - TScalar(1)) > kEpsilon<TScalar>) {
    x_dir_hint_foo.normalize();
  }
  if (abs(y_dir_hint_foo_sqr_length - TScalar(1)) > kEpsilon<TScalar>) {
    y_dir_hint_foo.normalize();
  }
  if (abs(normal_foo_sqr_length - TScalar(1)) > kEpsilon<TScalar>) {
    basis_foo.col(2).normalize();
  }

  TScalar abs_x_dot_z = abs(basis_foo.col(2).dot(x_dir_hint_foo));
  TScalar abs_y_dot_z = abs(basis_foo.col(2).dot(y_dir_hint_foo));
  if (abs_x_dot_z < abs_y_dot_z) {
    // basis_foo.z and xDirHint are far from parallel.
    basis_foo.col(1) = basis_foo.col(2).cross(x_dir_hint_foo).normalized();
    basis_foo.col(0) = basis_foo.col(1).cross(basis_foo.col(2));
  } else {
    // basis_foo.z and yDirHint are far from parallel.
    basis_foo.col(0) = y_dir_hint_foo.cross(basis_foo.col(2)).normalized();
    basis_foo.col(1) = basis_foo.col(2).cross(basis_foo.col(0));
  }
  TScalar det = basis_foo.determinant();
  // sanity check
  SOPHUS_ASSERT_NEAR(
      det,
      TScalar(1),
      kEpsilon<TScalar>,
      "Determinant of basis is not 1, but {}. Basis is \n{}\n",
      det,
      basis_foo);
  return basis_foo;
}

/// Takes in plane normal in reference frame foo and constructs a corresponding
/// rotation matrix ``foo_rotation_plane``.
///
/// See ``rotationFromNormal`` for details.
///
template <class TScalar>
auto rotation3FromPlane(Eigen::Vector3<TScalar> const& normal_in_foo)
    -> Rotation3<TScalar> {
  return Rotation3<TScalar>::fromRotationMatrix(
      rotation3FromNormal(normal_in_foo));
}

/// Returns a line (wrt. to frame ``foo``), given a pose of the ``line`` in
/// reference frame ``foo``.
///
/// Note: The plane is defined by X-axis of the ``line`` frame.
///
template <class TScalar>
auto lineFromIsometry(Isometry2<TScalar> const& foo_from_line)
    -> Eigen::Hyperplane<TScalar, 2> {
  return Eigen::Hyperplane<TScalar, 2>(
      normalFromRotation2(foo_from_line.rotation()),
      foo_from_line.translation());
}

/// Returns the pose ``T_foo_line``, given a line in reference frame ``foo``.
///
/// Note: The line is defined by X-axis of the frame ``line``.
///
template <class TScalar>
auto isometryFromLine(Eigen::Hyperplane<TScalar, 2> const& line_in_foo)
    -> Isometry2<TScalar> {
  TScalar const d = line_in_foo.offset();
  Eigen::Vector2<TScalar> const n = line_in_foo.normal();
  Rotation2<TScalar> const foo_rotation_plane = rotation2FromNormal(n);
  return Isometry2<TScalar>(-d * n, foo_rotation_plane);
}

/// Returns a plane (wrt. to frame ``foo``), given a pose of the ``plane`` in
/// reference frame ``foo``.
///
/// Note: The plane is defined by XY-plane of the frame ``plane``.
///
template <class TScalar>
auto planeFromIsometry(Isometry3<TScalar> const& foo_from_plane)
    -> Eigen::Hyperplane<TScalar, 3> {
  return Eigen::Hyperplane<TScalar, 3>(
      normalFromRotation3(foo_from_plane.so3()), foo_from_plane.translation());
}

/// Returns the pose ``foo_from_plane``, given a plane in reference frame
/// ``foo``.
///
/// Note: The plane is defined by XY-plane of the frame ``plane``.
///
template <class TScalar>
auto isometryFromPlane(Eigen::Hyperplane<TScalar, 3> const& plane_in_foo)
    -> Isometry3<TScalar> {
  TScalar const d = plane_in_foo.offset();
  Eigen::Vector3<TScalar> const n = plane_in_foo.normal();
  Rotation3<TScalar> const foo_rotation_plane = rotation3FromPlane(n);
  return Isometry3<TScalar>(-d * n, foo_rotation_plane);
}

/// Takes in a hyperplane and returns unique representation by ensuring that the
/// ``offset`` is not negative.
///
template <class TScalar, int kMatrixDim>
auto makeHyperplaneUnique(Eigen::Hyperplane<TScalar, kMatrixDim> const& plane)
    -> Eigen::Hyperplane<TScalar, kMatrixDim> {
  if (plane.offset() >= 0) {
    return plane;
  }

  return Eigen::Hyperplane<TScalar, kMatrixDim>(
      -plane.normal(), -plane.offset());
}

}  // namespace sophus
