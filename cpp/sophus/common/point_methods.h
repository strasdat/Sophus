
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/cast.h"
#include "sophus/common/point_concepts.h"
#include "sophus/common/reduce.h"

#include <Eigen/Core>

#include <algorithm>
#include <utility>
#include <vector>

namespace sophus {

namespace details {

// EigenDenseType may be a map or view or abstract base class or something.
// This is an alias for the corresponding concrete type with storage
template <EigenDenseType TPoint>
using EigenConcreteType = std::decay_t<decltype(std::declval<TPoint>().eval())>;

}  // namespace details

template <ScalarType TPoint>
[[nodiscard]] TPoint zero() {
  return 0;
}

template <EigenDenseType TPoint>
[[nodiscard]] TPoint zero() {
  return TPoint::Zero();
}

template <ScalarType TPoint>
auto eval(TPoint const& p) {
  return p;
}

template <EigenDenseType TPoint>
auto eval(TPoint const& p) {
  return p.eval();
}

template <ScalarType TPoint>
bool allTrue(TPoint const& p) {
  return bool(p);
}

template <EigenDenseType TPoint>
bool allTrue(TPoint const& p) {
  return p.all();
}

template <ScalarType TPoint>
bool anyTrue(TPoint const& p) {
  return bool(p);
}

template <EigenDenseType TPoint>
bool anyTrue(TPoint const& p) {
  return p.any();
}

template <ScalarType TPoint>
bool isFinite(TPoint const& p) {
  return std::isfinite(p);
}

template <EigenDenseType TPoint>
bool isFinite(TPoint const& p) {
  return p.array().isFinite().all();
}

template <ScalarType TPoint>
bool isNan(TPoint const& p) {
  return std::isnan(p);
}

template <PointType TPoint>
bool isNan(TPoint const& p) {
  return p.array().isNaN().all();
}

template <ScalarType TPoint>
auto square(TPoint const& v) {
  return v * v;
}

template <EigenDenseType TPoint>
auto square(TPoint const& v) {
  return v.squaredNorm();
}

template <ScalarType TPoint>
TPoint min(TPoint const& a, TPoint const& b) {
  return std::min(a, b);
}

template <EigenDenseType TPoint>
TPoint min(TPoint const& a, TPoint const& b) {
  return a.cwiseMin(b);
}

template <ScalarType TPoint>
TPoint max(TPoint const& a, TPoint const& b) {
  return std::max(a, b);
}

template <EigenDenseType TPoint>
TPoint max(TPoint const& a, TPoint const& b) {
  return a.cwiseMax(b);
}

template <PointType TPoint>
TPoint clamp(TPoint const& val, TPoint const& a, TPoint const& b) {
  return sophus::max(a, sophus::min(val, b));
}

template <ScalarType TPoint>
auto floor(TPoint s) {
  using std::floor;
  return floor(s);
}

template <EigenDenseType TPoint>
auto floor(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::floor(e);
  }
  return p;
}

template <ScalarType TPoint>
auto ceil(TPoint s) {
  using std::ceil;
  return ceil(s);
}

template <EigenDenseType TPoint>
auto ceil(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::ceil(e);
  }
  return p;
}

template <ScalarType TPoint>
auto round(TPoint s) {
  using std::ceil;
  return ceil(s);
}

template <EigenDenseType TPoint>
auto round(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::round(e);
  }
  return p;
}

template <ScalarType TPoint>
[[nodiscard]] auto plus(TPoint p, TPoint s) {
  p += s;
  return p;
}

template <EigenDenseType TPoint>
[[nodiscard]] auto plus(TPoint p, typename TPoint::Scalar s) {
  p.array() += s;
  return p;
}

template <ScalarType TPoint>
[[nodiscard]] bool isLessEqual(TPoint const& lhs, TPoint const& rhs) {
  return lhs <= rhs;
}

template <EigenDenseType TPoint>
[[nodiscard]] bool isLessEqual(TPoint const& lhs, TPoint const& rhs) {
  return allTrue(eval(lhs.array() <= rhs.array()));
}

template <ScalarType TPoint>
[[nodiscard]] Expected<TPoint> tryGetElem(
    TPoint const& p, size_t row, size_t col = 0) {
  if (row == 0 && col == 0) {
    return p;
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <EigenDenseType TPoint>
[[nodiscard]] Expected<TPoint> tryGetElem(
    TPoint const& p, size_t row, size_t col = 0) {
  if (row < p.rows() && col < p.cols()) {
    return p(row, col);
  }
  return SOPHUS_UNEXPECTED(
      "({}, {}) access of array of size {} x {}", row, col, p.rows(), p.cols());
}

template <ScalarType TPoint>
[[nodiscard]] Expected<Success> trySetElem(
    TPoint& p, TPoint s, size_t row, size_t col = 0) {
  if (row == 0 && col == 0) {
    p = s;
    return Success{};
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <EigenDenseType TPoint>
[[nodiscard]] Expected<Success> trySetElem(
    TPoint& p, typename TPoint::Scalar s, size_t row, size_t col = 0) {
  if (row == 0 && col == 0) {
    p(row, col) = s;
    return Success{};
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <ScalarType TPoint>
[[nodiscard]] auto transpose(TPoint p) {
  return p;
}

template <EigenDenseType TPoint>
[[nodiscard]] auto transpose(TPoint p) {
  return p.transpose();
}

}  // namespace sophus
