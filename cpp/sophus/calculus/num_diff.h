// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Numerical differentiation using finite differences

#pragma once

#include "sophus/common/common.h"

#include <functional>
#include <type_traits>
#include <utility>

namespace sophus {

namespace details {
template <class TScalar>
class Curve {
 public:
  template <class TFn>
  static auto numDiff(TFn curve, TScalar t, TScalar h) -> decltype(curve(t)) {
    static_assert(
        std::is_floating_point<TScalar>::value,
        "Scalar must be a floating point type.");

    return (curve(t + h) - curve(t - h)) / (TScalar(2) * h);
  }
};

template <class TScalar, int kMatrixDim, int kM>
class VectorField {
 public:
  static auto numDiff(
      std::function<Eigen::Vector<TScalar, kMatrixDim>(
          Eigen::Vector<TScalar, kM>)> vector_field,
      Eigen::Vector<TScalar, kM> const& a,
      TScalar eps) -> Eigen::Matrix<TScalar, kMatrixDim, kM> {
    static_assert(
        std::is_floating_point<TScalar>::value,
        "Scalar must be a floating point type.");
    Eigen::Matrix<TScalar, kMatrixDim, kM> j;
    Eigen::Vector<TScalar, kM> h;
    h.setZero();
    for (int i = 0; i < kM; ++i) {
      h[i] = eps;
      Eigen::Vector<TScalar, kMatrixDim> vfp = vector_field(a + h);
      Eigen::Vector<TScalar, kMatrixDim> vfm = vector_field(a - h);

      j.col(i) = (vfp - vfm) / (TScalar(2) * eps);
      h[i] = TScalar(0);
    }

    return j;
  }
};

}  // namespace details

/// Calculates the derivative of a curve at a point ``t``.
///
/// Here, a curve is a function from a Scalar to a Euclidean space. Thus, it
/// returns either a Scalar, a vector or a matrix.
///
template <class TScalar, class TFn>
auto curveNumDiff(TFn curve, TScalar t, TScalar h = kEpsilonSqrt<TScalar>)
    -> decltype(details::Curve<TScalar>::numDiff(std::move(curve), t, h)) {
  return details::Curve<TScalar>::numDiff(std::move(curve), t, h);
}

/// Calculates the derivative of a vector field at a point ``a``.
///
/// Here, a vector field is a function from a vector space to another vector
/// space.
///
template <
    class TScalar,
    int kMatrixDim,
    int kM,
    class TScalarOrVector,
    class TFn>
auto vectorFieldNumDiff(
    TFn vector_field,
    TScalarOrVector const& a,
    TScalar eps = kEpsilonSqrt<TScalar>)
    -> Eigen::Matrix<TScalar, kMatrixDim, kM> {
  return details::VectorField<TScalar, kMatrixDim, kM>::numDiff(
      vector_field, a, eps);
}

}  // namespace sophus
