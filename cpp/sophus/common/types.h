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

#include "sophus/common/common.h"

#include <type_traits>

namespace sophus {

namespace details {

struct UninitTag {};

template <class TScalar>
class MaxMetric {
 public:
  static TScalar impl(TScalar s0, TScalar s1) {
    using std::abs;
    return abs(s0 - s1);
  }
};

template <class TScalar, int kM, int kMatrixDim>
class MaxMetric<Eigen::Matrix<TScalar, kM, kMatrixDim>> {
 public:
  static TScalar impl(
      Eigen::Matrix<TScalar, kM, kMatrixDim> const& p0,
      Eigen::Matrix<TScalar, kM, kMatrixDim> const& p1) {
    return (p0 - p1).template lpNorm<Eigen::Infinity>();
  }
};

template <class TScalar>
class SetToZero {
 public:
  static void impl(TScalar& s) { s = TScalar(0); }
};

template <class TScalar, int kM, int kMatrixDim>
class SetToZero<Eigen::Matrix<TScalar, kM, kMatrixDim>> {
 public:
  static void impl(Eigen::Matrix<TScalar, kM, kMatrixDim>& v) { v.setZero(); }
};

template <class TT, class TScalar>
class SetElementAt;

template <class TScalar>
class SetElementAt<TScalar, TScalar> {
 public:
  static void impl(TScalar& s, TScalar value, int at) {
    FARM_CHECK(at == 0, "is {}", at);
    s = value;
  }
};

template <class TScalar, int kMatrixDim>
class SetElementAt<Eigen::Vector<TScalar, kMatrixDim>, TScalar> {
 public:
  static void impl(
      Eigen::Vector<TScalar, kMatrixDim>& v, TScalar value, int at) {
    FARM_CHECK(at >= 0 && at < kMatrixDim, "is {}", at);
    v[at] = value;
  }
};

template <class TScalar>
class SquaredNorm {
 public:
  static TScalar impl(TScalar const& s) { return s * s; }
};

template <class TScalar, int kMatrixDim>
class SquaredNorm<Eigen::Matrix<TScalar, kMatrixDim, 1>> {
 public:
  static TScalar impl(Eigen::Matrix<TScalar, kMatrixDim, 1> const& s) {
    return s.squaredNorm();
  }
};

template <class TScalar>
class Transpose {
 public:
  static TScalar impl(TScalar const& s) { return s; }
};

template <class TScalar, int kM, int kMatrixDim>
class Transpose<Eigen::Matrix<TScalar, kM, kMatrixDim>> {
 public:
  static Eigen::Matrix<TScalar, kM, kMatrixDim> impl(
      Eigen::Matrix<TScalar, kM, kMatrixDim> const& s) {
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
template <class TT, class TScalar>
void setElementAt(TT& p, TScalar value, int i) {
  return details::SetElementAt<TT, TScalar>::impl(p, value, i);
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

template <class TScalar>
struct IsFloatingPoint {
  static bool const kValue = std::is_floating_point<TScalar>::value;
};

template <class TScalar, int kM, int kMatrixDim>
struct IsFloatingPoint<Eigen::Matrix<TScalar, kM, kMatrixDim>> {
  static bool const kValue = std::is_floating_point<TScalar>::value;
};

template <class TScalar>
struct GetScalar {
  using Scalar = TScalar;
};

template <class TScalar, int kM, int kMatrixDim>
struct GetScalar<Eigen::Matrix<TScalar, kM, kMatrixDim>> {
  using Scalar = TScalar;
};

/// If the Vector type is of fixed size, then IsFixedSizeVector::value will be
/// true.
template <
    typename TVectorT,
    int kNumDimensions,
    typename = typename std::enable_if<
        TVectorT::RowsAtCompileTime == kNumDimensions &&
        TVectorT::ColsAtCompileTime == 1>::type>
struct IsFixedSizeVector : std::true_type {};

}  // namespace sophus
