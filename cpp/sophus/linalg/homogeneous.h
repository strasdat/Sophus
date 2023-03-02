// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"

namespace sophus {

/// Projects 3-point (x,y,z) through the origin (0,0,0) onto the plane z=1.
/// Hence it returns (x/z, y/z).
///
/// Precondition: z must not be close to 0.
template <class TPoint>
auto proj(Eigen::MatrixBase<TPoint> const& p)
    -> Eigen::Vector<typename TPoint::Scalar, TPoint::RowsAtCompileTime - 1> {
  static_assert(TPoint::ColsAtCompileTime == 1, "p must be a column-vector");
  static_assert(TPoint::RowsAtCompileTime >= 2, "p must have at least 2 rows");
  return p.template head<TPoint::RowsAtCompileTime - 1>() /
         p[TPoint::RowsAtCompileTime - 1];
}

/// Maps point on the z=1 plane (a,b) to homogeneous representation of the same
/// point: (z*a, z*b, z). Z defaults to 1.
template <class TPoint>
auto unproj(
    Eigen::MatrixBase<TPoint> const& p, const typename TPoint::Scalar& z = 1.0)
    -> Eigen::Vector<typename TPoint::Scalar, TPoint::RowsAtCompileTime + 1> {
  using Scalar = typename TPoint::Scalar;
  static_assert(TPoint::ColsAtCompileTime == 1, "p must be a column-vector");
  Eigen::Vector<Scalar, TPoint::RowsAtCompileTime + 1> out;
  out.template head<TPoint::RowsAtCompileTime>() = z * p;
  out[TPoint::RowsAtCompileTime] = z;
  return out;
}

}  // namespace sophus
