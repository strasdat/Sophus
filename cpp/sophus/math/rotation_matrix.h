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

#include "sophus/core/types.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace sophus {

/// Takes in arbitrary square matrix and returns true if it is
/// orthogonal.
template <class DT>
SOPHUS_FUNC bool isOrthogonal(Eigen::MatrixBase<DT> const& r) {
  using Scalar = typename DT::Scalar;
  static int const kMatrixDim = DT::RowsAtCompileTime;
  static int const kM = DT::ColsAtCompileTime;

  static_assert(kMatrixDim == kM, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  return (r * r.transpose() -
          Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity())
             .norm() < kEpsilon<Scalar>;
}

/// Takes in arbitrary square matrix and returns true if it is
/// "scaled-orthogonal" with positive determinant.
///
template <class DT>
SOPHUS_FUNC bool isScaledOrthogonalAndPositive(
    Eigen::MatrixBase<DT> const& s_r) {
  using Scalar = typename DT::Scalar;
  static int const kMatrixDim = DT::RowsAtCompileTime;
  static int const kM = DT::ColsAtCompileTime;
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
template <class DT>
SOPHUS_FUNC std::enable_if_t<
    std::is_floating_point<typename DT::Scalar>::value,
    Eigen::Matrix<
        typename DT::Scalar,
        DT::RowsAtCompileTime,
        DT::RowsAtCompileTime>>
makeRotationMatrix(Eigen::MatrixBase<DT> const& r) {
  using Scalar = typename DT::Scalar;
  static int const kMatrixDim = DT::RowsAtCompileTime;
  static int const kM = DT::ColsAtCompileTime;

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
