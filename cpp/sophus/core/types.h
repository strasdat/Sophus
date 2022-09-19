// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Common type aliases.

#pragma once

#include "sophus/core/common.h"

#include <type_traits>

namespace sophus {

namespace details {
template <class ScalarT>
class MaxMetric {
 public:
  static ScalarT impl(ScalarT s0, ScalarT s1) {
    using std::abs;
    return abs(s0 - s1);
  }
};

template <class ScalarT, int kM, int kMatrixDim>
class MaxMetric<Eigen::Matrix<ScalarT, kM, kMatrixDim>> {
 public:
  static ScalarT impl(
      Eigen::Matrix<ScalarT, kM, kMatrixDim> const& p0,
      Eigen::Matrix<ScalarT, kM, kMatrixDim> const& p1) {
    return (p0 - p1).template lpNorm<Eigen::Infinity>();
  }
};

template <class ScalarT>
class SetToZero {
 public:
  static void impl(ScalarT& s) { s = ScalarT(0); }
};

template <class ScalarT, int kM, int kMatrixDim>
class SetToZero<Eigen::Matrix<ScalarT, kM, kMatrixDim>> {
 public:
  static void impl(Eigen::Matrix<ScalarT, kM, kMatrixDim>& v) { v.setZero(); }
};

template <class T1T, class ScalarT>
class SetElementAt;

template <class ScalarT>
class SetElementAt<ScalarT, ScalarT> {
 public:
  static void impl(ScalarT& s, ScalarT value, int at) {
    FARM_CHECK(at == 0, "is {}", at);
    s = value;
  }
};

template <class ScalarT, int kMatrixDim>
class SetElementAt<Eigen::Vector<ScalarT, kMatrixDim>, ScalarT> {
 public:
  static void impl(
      Eigen::Vector<ScalarT, kMatrixDim>& v, ScalarT value, int at) {
    FARM_CHECK(at >= 0 && at < kMatrixDim, "is {}", at);
    v[at] = value;
  }
};

template <class ScalarT>
class SquaredNorm {
 public:
  static ScalarT impl(ScalarT const& s) { return s * s; }
};

template <class ScalarT, int kMatrixDim>
class SquaredNorm<Eigen::Matrix<ScalarT, kMatrixDim, 1>> {
 public:
  static ScalarT impl(Eigen::Matrix<ScalarT, kMatrixDim, 1> const& s) {
    return s.squaredNorm();
  }
};

template <class ScalarT>
class Transpose {
 public:
  static ScalarT impl(ScalarT const& s) { return s; }
};

template <class ScalarT, int kM, int kMatrixDim>
class Transpose<Eigen::Matrix<ScalarT, kM, kMatrixDim>> {
 public:
  static Eigen::Matrix<ScalarT, kM, kMatrixDim> impl(
      Eigen::Matrix<ScalarT, kM, kMatrixDim> const& s) {
    return s.transpose().eval();
  }
};
}  // namespace details

/// Returns maximum metric between two points ``p0`` and ``p1``, with ``p0, p1``
/// being matrices or a scalars.
///
template <class TT>
auto maxMetric(TT const& p0, TT const& p1)
    -> decltype(details::MaxMetric<TT>::impl(p0, p1)) {
  return details::MaxMetric<TT>::impl(p0, p1);
}

/// Sets point ``p`` to zero, with ``p`` being a matrix or a scalar.
///
template <class TT>
void setToZero(TT& p) {
  return details::SetToZero<TT>::impl(p);
}

/// Sets ``i``th component of ``p`` to ``value``, with ``p`` being a
/// matrix or a scalar. If ``p`` is a scalar, ``i`` must be ``0``.
///
template <class TT, class ScalarT>
void setElementAt(TT& p, ScalarT value, int i) {
  return details::SetElementAt<TT, ScalarT>::impl(p, value, i);
}

/// Returns the squared 2-norm of ``p``, with ``p`` being a vector or a scalar.
///
template <class TT>
auto squaredNorm(TT const& p) -> decltype(details::SquaredNorm<TT>::impl(p)) {
  return details::SquaredNorm<TT>::impl(p);
}

/// Returns ``p.transpose()`` if ``p`` is a matrix, and simply ``p`` if m is a
/// scalar.
///
template <class TT>
auto transpose(TT const& p) -> decltype(details::Transpose<TT>::impl(TT())) {
  return details::Transpose<TT>::impl(p);
}

template <class ScalarT>
struct IsFloatingPoint {
  static bool const kValue = std::is_floating_point<ScalarT>::value;
};

template <class ScalarT, int kM, int kMatrixDim>
struct IsFloatingPoint<Eigen::Matrix<ScalarT, kM, kMatrixDim>> {
  static bool const kValue = std::is_floating_point<ScalarT>::value;
};

template <class ScalarT>
struct GetScalar {
  using Scalar = ScalarT;
};

template <class ScalarT, int kM, int kMatrixDim>
struct GetScalar<Eigen::Matrix<ScalarT, kM, kMatrixDim>> {
  using Scalar = ScalarT;
};

/// If the Vector type is of fixed size, then IsFixedSizeVector::value will be
/// true.
template <
    typename VectorT,
    int kNumDimensions,
    typename = typename std::enable_if<
        VectorT::RowsAtCompileTime == kNumDimensions &&
        VectorT::ColsAtCompileTime == 1>::type>
struct IsFixedSizeVector : std::true_type {};

}  // namespace sophus
