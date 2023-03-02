// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/geometry/inverse_depth.h"
#include "sophus/lie/se3.h"
#include "sophus/sensor/camera_projection/projection_z1.h"

namespace sophus {

/// Projects 3-point (a,b,psi) = (x/z,y/z,1/z) through the origin (0,0,0) onto
/// the plane z=1. Hence it returns (a,b) = (x/z, y/z).
template <class TT>
auto proj(InverseDepthPoint3<TT> const& inverse_depth_point)
    -> Eigen::Matrix<TT, 2, 1> {
  return inverse_depth_point.projInZ1Plane();
}

/// Returns point derivative of inverse depth point projection:
///
///   Dx proj(x) with x = (a,b,psi) being an inverse depth point.
template <class TT>
auto dxProjX(InverseDepthPoint3<TT> const& /*inverse_depth_point*/)
    -> Eigen::Matrix<TT, 2, 3> {
  Eigen::Matrix<TT, 2, 3> dx;
  dx.setIdentity();
  return dx;
}

/// Returns pose derivative of inverse depth point projection at the identity:
///
///   Dx proj(exp(x) * y) at x=0
///
/// with y = (a,b,psi) being an inverse depth point.
template <class TT>
auto dxProjExpXPointAt0(InverseDepthPoint3<TT> const& inverse_depth_point)
    -> Eigen::Matrix<TT, 2, 6> {
  Eigen::Matrix<TT, 2, 6> dx;
  TT i = TT(1);
  TT psi = inverse_depth_point.psi();

  TT a = inverse_depth_point.projInZ1Plane().x();
  TT b = inverse_depth_point.projInZ1Plane().y();
  // clang-format off
  dx <<
    psi,   0, -psi * a,     -a*b, a*a + i, -b,
      0, psi, -psi * b, -b*b - i,     a*b,  a;
  // clang-format on
  return dx;
}

/// Transforms inverse_depth point in frame bar to a scaled inverse depth point
/// in frame foo. Here the scale is psi, the input inverse depth.
///
/// Given (a,b,psi) being the inverse depth point in frame bar, it returns
///
///   psi * (foo_from_bar * inverse_depth_point_in_bar.toEuclideanPoint3())
///
/// for psi!=0.
template <class TT>
auto scaledTransform(
    sophus::Isometry3<TT> const& foo_from_bar,
    InverseDepthPoint3<TT> const& inverse_depth_point_in_bar)
    -> Eigen::Matrix<TT, 3, 1> {
  return foo_from_bar.so3() *
             unproj(inverse_depth_point_in_bar.projInZ1Plane()) +
         inverse_depth_point_in_bar.psi() * foo_from_bar.translation();
}

/// Transforms inverse_depth point from frame bar to frame foo followed by a
/// projection.
///
/// If psi != 0, hence the point is not at +/- infinity, this function is
/// equivalent to:
///
///   camProj(foo_from_bar * inverse_depth_point_in_bar.toEuclideanPoint3());
///
/// However, this function can also applied when 1/z==0, hence the point is at
/// +/- infinity.
template <class TT>
auto projTransform(
    sophus::Isometry3<TT> const& foo_from_bar,
    InverseDepthPoint3<TT> const& inverse_depth_point_in_bar)
    -> Eigen::Matrix<TT, 2, 1> {
  //      R * (x,y,z) + t
  //   =  z * [R * (x/z, y/z, 1) + 1/z * t]
  //
  // Hence:
  //      proj(R * (x,y,z) + t)
  //   =  proj(1/z * [R * (x,y,z) + t])     { since proj(xyz) == proj(l * xyz) }
  //   =  proj(R * (x/z, y/z, 1) + 1/z * t)
  //
  // qed.
  //       with R := foo_from_bar.so3(),
  //            t := foo_from_bar.translation()
  //            (x/z, y/z, 1) := unproj(rojInZ1Plane())
  //            1/z := psi()
  //
  return proj(scaledTransform(foo_from_bar, inverse_depth_point_in_bar));
}

/// Functor to efficiently transform a number of point given a Isometry3 pose.
///
/// When transforming a point `point_in_bar` given a sophus::Isometry3 pose
/// `foo_from_bar`, one can simply use
///
///   ``Eigen::Vector3d  = foo_from_bar * point_in_bar;``
///
/// Internally, this applies the (unit) quaternion to the left and the right of
/// the point to rotate it and then adds the translation:
///
///     point_in_foo = q*point_in_bar*q' + bar_origin_in_foo.
///
/// If there are a lot of point to transform, there is a more efficient
/// way using the rotation matrix R which can be precomputed from the quaternion
/// q.
///
///    point_in_foo = R * point_in_bar + bar_origin_in_foo
///
/// This is what this functor is for.
template <class TT>
class PointTransformer {
 public:
  PointTransformer() = default;
  explicit PointTransformer(sophus::Isometry3<TT> const& foo_from_bar)
      : foo_from_bar_(foo_from_bar),
        foo_rotation_bar_(foo_from_bar.so3().matrix()),
        bar_origin_in_foo_(foo_from_bar.translation()) {}

  /// Transforms a 3-point from frame bar to frame foo.
  [[nodiscard]] auto transform(Eigen::Matrix<TT, 3, 1> const& point_in_bar)
      const -> Eigen::Matrix<TT, 3, 1> {
    return foo_rotation_bar_ * point_in_bar + bar_origin_in_foo_;
  }

  [[nodiscard]] auto scaledTransform(
      InverseDepthPoint3<TT> const& inverse_depth_point_in_bar) const
      -> Eigen::Matrix<TT, 3, 1> {
    return foo_rotation_bar_ *
               unproj(inverse_depth_point_in_bar.projInZ1Plane()) +
           inverse_depth_point_in_bar.psi() * bar_origin_in_foo_;
  }

  /// Transforms 3-point in frame bar to foo and projects it onto the
  /// Euclidean plane z=1 in foo.
  [[nodiscard]] auto projTransform(Eigen::Matrix<TT, 3, 1> const& point_in_bar)
      const -> Eigen::Matrix<TT, 2, 1> {
    return proj(foo_rotation_bar_ * point_in_bar + bar_origin_in_foo_);
  }

  /// Transforms and projects the 3d inverse depth point in frame bar to the
  /// Euclidean plane z=1 in foo.
  [[nodiscard]] auto projTransform(
      InverseDepthPoint3<TT> const& inverse_depth_point_in_bar) const
      -> Eigen::Matrix<TT, 2, 1> {
    return proj(scaledTransform(inverse_depth_point_in_bar));
  }

  /// Returns pose derivative of inverse depth point projection at the identity:
  ///
  ///   Dx proj(exp(x) * foo_from_bar * foo_in_bar.toEuclideanPoint3()) at x=0
  ///
  /// with foo_in_bar = (a,b,psi) being an inverse depth point.
  [[nodiscard]] auto dxProjExpXTransformPointAt0(
      InverseDepthPoint3<TT> const& inverse_depth_point_in_bar) const
      -> Eigen::Matrix<TT, 2, 6> {
    Eigen::Matrix<TT, 2, 6> dx;
    TT i = TT(1);
    TT psi = inverse_depth_point_in_bar.psi();
    Eigen::Matrix<TT, 3, 1> scaled_point_in_foo =
        scaledTransform(inverse_depth_point_in_bar);
    TT x = scaled_point_in_foo[0];
    TT y = scaled_point_in_foo[1];
    TT z = scaled_point_in_foo[2];
    TT z_sq = z * z;

    // clang-format off
    dx <<
      psi/z,     0, -psi * x / z_sq,     -x*y/z_sq, x*x/z_sq + i, -y/z,
          0, psi/z, -psi * y / z_sq, -y*y/z_sq - i,     x*y/z_sq,  x/z;
    // clang-format on
    return dx;
  }

  // Assuming ProjectionZ1 based camera
  [[nodiscard]] auto dxProjTransformX(
      InverseDepthPoint3<TT> const& inverse_depth_point_in_bar) const
      -> Eigen::Matrix<TT, 2, 3> {
    Eigen::Vector3<TT> const& r0 = this->fooRotationBar().col(0);
    Eigen::Vector3<TT> const& r1 = this->fooRotationBar().col(1);
    Eigen::Vector3<TT> const& t = this->barOriginInFoo();

    Eigen::Matrix3<TT> mat_j;
    mat_j << r0, r1, t;

    return ProjectionZ1::dxProjX(scaledTransform(inverse_depth_point_in_bar)) *
           mat_j;
  }

  [[nodiscard]] auto fooFromBar() const -> sophus::Isometry3<TT> const& {
    return foo_from_bar_;
  }
  [[nodiscard]] auto fooRotationBar() const -> Eigen::Matrix<TT, 3, 3> const& {
    return foo_rotation_bar_;
  }
  [[nodiscard]] auto barOriginInFoo() const -> Eigen::Matrix<TT, 3, 1> const& {
    return bar_origin_in_foo_;
  }

 private:
  sophus::Isometry3<TT> foo_from_bar_;
  Eigen::Matrix<TT, 3, 3> foo_rotation_bar_;
  Eigen::Matrix<TT, 3, 1> bar_origin_in_foo_;
};
}  // namespace sophus
