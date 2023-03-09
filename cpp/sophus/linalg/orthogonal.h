// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Rotation matrix helper functions.

#pragma once

#include "sophus/common/common.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace sophus {

/// Takes in arbitrary square matrix and returns true if it is
/// orthogonal.
template <class TD>
auto isOrthogonal(Eigen::MatrixBase<TD> const& r) -> bool {
  using Scalar = typename TD::Scalar;
  static int const kMatrixDim = TD::RowsAtCompileTime;
  static int const kM = TD::ColsAtCompileTime;

  static_assert(kMatrixDim == kM, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  return (r * r.transpose() -
          Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity())
             .norm() < kEpsilonSqrt<Scalar>;
}

/// Takes in arbitrary square matrix and returns true if it is
/// "scaled-orthogonal" with positive determinant.
///
template <class TD>
auto isScaledOrthogonalAndPositive(Eigen::MatrixBase<TD> const& s_r) -> bool {
  using Scalar = typename TD::Scalar;
  static int const kMatrixDim = TD::RowsAtCompileTime;
  static int const kM = TD::ColsAtCompileTime;
  using std::pow;
  using std::sqrt;

  Scalar det = s_r.determinant();

  if (det <= Scalar(0)) {
    return false;
  }

  Scalar scale_sqr = pow(det, Scalar(2. / kMatrixDim));

  static_assert(kMatrixDim == kM, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  return (s_r * s_r.transpose() -
          scale_sqr * Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity())
             .template lpNorm<Eigen::Infinity>() < sqrt(kEpsilon<Scalar>);
}

/// Takes in arbitrary square matrix (2x2 or larger) and returns closest
/// orthogonal matrix with positive determinant.
template <class TD>
auto makeRotationMatrix(Eigen::MatrixBase<TD> const& r) -> std::enable_if_t<
    std::is_floating_point<typename TD::Scalar>::value,
    Eigen::Matrix<
        typename TD::Scalar,
        TD::RowsAtCompileTime,
        TD::RowsAtCompileTime>> {
  using Scalar = typename TD::Scalar;
  static int const kMatrixDim = TD::RowsAtCompileTime;
  static int const kM = TD::ColsAtCompileTime;

  static_assert(kMatrixDim == kM, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  Eigen::JacobiSVD<Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>> svd(
      r, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Determine determinant of orthogonal matrix U*V'.
  Scalar d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  // Starting from the identity matrix D, set the last entry to d (+1 or
  // -1),  so that det(U*D*V') = 1.
  Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim> diag =
      Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity();
  diag(kMatrixDim - 1, kMatrixDim - 1) = d;
  return svd.matrixU() * diag * svd.matrixV().transpose();
}

}  // namespace sophus
