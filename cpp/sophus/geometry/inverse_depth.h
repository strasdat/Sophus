// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/geometry/projection.h"

namespace sophus {

/// Inverse depth point representation
///
///   (a, b) := (x/z, y/z) and psi := 1/z
///
/// following https://ethaneade.com/thesis_revised.pdf, pp. 79
///
/// Let us assume we have Euclidean 3d point (x,y,z) in a local reference frame
/// (e.g. camera origin). One can construct an inverse depth point (in the same
/// local reference frame) as follows:
///
///   First we project the point (x,y,z) through the origin (0,0,0) onto the z=1
///   plane. We call the projection (a, b) := (x/z, y/z).
///
/// In other words, (a, b) is the intersection of the line through (0, 0, 0)
/// to (x, y, z) and the 2d Euclidean plane z=1.
///
/// Now, we can describe almost any 3d point in our local reference frame as a
/// point (a',b') in the Euclidean plane z=1 and the inverse depth psi := 1/z.
///
/// For example, the Euclidean point (2, 0, 8) is represented as
/// (a, b) = (2/8, 0/8) = (1/4, 0) and inverse depth psi = 1/8.
///
/// The only Euclidean 3d point we cannot describe is the origin (0,0,0) (since
/// there are infinitely many lines through the origin which intersect with the
/// plane z=1).
///
/// The advantage of using an inverse depth representation over Euclidean
/// representation is that we can also represent points at infinity.
/// Let (a,b) a 2d Euclidean point on our reference plane z=1; nothing stops us
/// from choosing a psi=0, which corresponds to a "z = 1/0 = infinity".
///
/// In summary, using this representation, we can represent
///  - points at infinity:             psi == 1/z == 0
///  - points close to +infinity:      psi == 1/z == +e
///  - points close to -infinity:      psi == 1/z == -e
///  - points one unit in front:       psi == 1/z == +1
///  - points one unit behind:         psi == 1/z == -1
///  - points close to zero, in front: psi == 1/z == +999999
///  - points close to zero, behind:   psi == 1/z == -999999
///
template <class T>
class InverseDepthPoint3 {
 public:
  InverseDepthPoint3() {}

  static InverseDepthPoint3 fromEuclideanPoint3(
      Eigen::Matrix<T, 3, 1> const& p) {
    using std::abs;
    SOPHUS_ASSERT_GE(abs(p.z()), sophus::kEpsilon<T>);
    return InverseDepthPoint3(p.x() / p.z(), p.y() / p.z(), 1.0 / p.z());
  }

  static InverseDepthPoint3 fromAbAndPsi(
      Eigen::Matrix<T, 3, 1> const& ab_and_psi) {
    InverseDepthPoint3 p;
    p.ab_and_psi_ = ab_and_psi;
    return p;
  }

  InverseDepthPoint3(
      Eigen::Matrix<T, 2, 1> const& proj_in_z1_plane, T const& one_by_z)
      : ab_and_psi_(proj_in_z1_plane[0], proj_in_z1_plane[1], one_by_z) {
    SOPHUS_ASSERT_GE(ab_and_psi_.norm(), sophus::kEpsilon<T>);
  }

  InverseDepthPoint3(T const& x_by_z, T const& y_by_z, T const& one_by_z)
      : ab_and_psi_(x_by_z, y_by_z, one_by_z) {}

  // Returns the projection of the point (x,y,z) onto the plane z=1.
  // Hence (a, b) = ("x / z", "y / z").
  [[nodiscard]] Eigen::Matrix<T, 2, 1> projInZ1Plane() const {
    return ab_and_psi_.template head<2>();
  }

  /// Returns inverse depth psi, hence "1 / z".
  [[nodiscard]] T const& psi() const { return ab_and_psi_[2]; }
  T& psi() { return ab_and_psi_[2]; }

  [[nodiscard]] T const* data() const { return ab_and_psi_.data(); }

  T* data() { return ab_and_psi_.data(); }

  [[nodiscard]] [[nodiscard]] Eigen::Matrix<T, 3, 1> const& params() const {
    return ab_and_psi_;
  }

  /// Precondition: psi must not be close to 0, hence z must not be near
  /// infinity.
  [[nodiscard]] Eigen::Matrix<T, 3, 1> toEuclideanPoint3() const {
    using std::abs;
    SOPHUS_ASSERT_GE(abs(psi()), sophus::kEpsilon<T>);

    return Eigen::Matrix<T, 3, 1>(
        ab_and_psi_.x() / psi(), ab_and_psi_.y() / psi(), T(1) / psi());
  }

 private:
  Eigen::Matrix<T, 3, 1> ab_and_psi_;
};

using InverseDepthPoint3F64 = InverseDepthPoint3<double>;

}  // namespace sophus
