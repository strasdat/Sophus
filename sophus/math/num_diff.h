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

#include "sophus/core/types.h"

#include <functional>
#include <type_traits>
#include <utility>

namespace sophus {

namespace details {
template <class ScalarT>
class Curve {
 public:
  template <class FnT>
  static auto numDiff(FnT curve, ScalarT t, ScalarT h) -> decltype(curve(t)) {
    using ReturnType = decltype(curve(t));
    static_assert(
        std::is_floating_point<ScalarT>::value,
        "Scalar must be a floating point type.");
    static_assert(
        IsFloatingPoint<ReturnType>::kValue,
        "ReturnType must be either a floating point scalar, "
        "vector or matrix.");

    return (curve(t + h) - curve(t - h)) / (ScalarT(2) * h);
  }
};

template <class ScalarT, int kMatrixDim, int kM>
class VectorField {
 public:
  static Eigen::Matrix<ScalarT, kMatrixDim, kM> numDiff(
      std::function<Eigen::Vector<ScalarT, kMatrixDim>(
          Eigen::Vector<ScalarT, kM>)> vector_field,
      Eigen::Vector<ScalarT, kM> const& a,
      ScalarT eps) {
    static_assert(
        std::is_floating_point<ScalarT>::value,
        "Scalar must be a floating point type.");
    Eigen::Matrix<ScalarT, kMatrixDim, kM> j;
    Eigen::Vector<ScalarT, kM> h;
    h.setZero();
    for (int i = 0; i < kM; ++i) {
      h[i] = eps;
      j.col(i) =
          (vector_field(a + h) - vector_field(a - h)) / (ScalarT(2) * eps);
      h[i] = ScalarT(0);
    }

    return j;
  }
};

template <class ScalarT, int kMatrixDim>
class VectorField<ScalarT, kMatrixDim, 1> {
 public:
  static Eigen::Matrix<ScalarT, kMatrixDim, 1> numDiff(
      std::function<Eigen::Vector<ScalarT, kMatrixDim>(ScalarT)> vector_field,
      ScalarT const& a,
      ScalarT eps) {
    return details::Curve<ScalarT>::numDiff(std::move(vector_field), a, eps);
  }
};
}  // namespace details

/// Calculates the derivative of a curve at a point ``t``.
///
/// Here, a curve is a function from a Scalar to a Euclidean space. Thus, it
/// returns either a Scalar, a vector or a matrix.
///
template <class ScalarT, class FnT>
auto curveNumDiff(FnT curve, ScalarT t, ScalarT h = kEpsilonSqrt<ScalarT>)
    -> decltype(details::Curve<ScalarT>::numDiff(std::move(curve), t, h)) {
  return details::Curve<ScalarT>::numDiff(std::move(curve), t, h);
}

/// Calculates the derivative of a vector field at a point ``a``.
///
/// Here, a vector field is a function from a vector space to another vector
/// space.
///
template <
    class ScalarT,
    int kMatrixDim,
    int kM,
    class ScalarOrVectorT,
    class FnT>
Eigen::Matrix<ScalarT, kMatrixDim, kM> vectorFieldNumDiff(
    FnT vector_field,
    ScalarOrVectorT const& a,
    ScalarT eps = kEpsilonSqrt<ScalarT>) {
  return details::VectorField<ScalarT, kMatrixDim, kM>::numDiff(
      std::move(vector_field), a, eps);
}

}  // namespace sophus
