
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/linalg/cast.h"
#include "sophus/linalg/reduce.h"

#include <Eigen/Core>

#include <algorithm>
#include <utility>
#include <vector>

namespace sophus {

namespace details {

// EigenDenseType may be a map or view or abstract base class or something.
// This is an alias for the corresponding concrete type with storage
template <::sophus::concepts::EigenDenseType TPoint>
using EigenConcreteType = std::decay_t<decltype(std::declval<TPoint>().eval())>;

}  // namespace details

template <::sophus::concepts::ScalarType TPoint>
[[nodiscard]] auto zero() -> TPoint {
  return 0;
}

template <::sophus::concepts::EigenDenseType TPoint>
[[nodiscard]] auto zero() -> TPoint {
  return TPoint::Zero();
}

template <::sophus::concepts::ScalarType TPoint>
auto eval(TPoint const& p) {
  return p;
}

template <::sophus::concepts::EigenDenseType TPoint>
auto eval(TPoint const& p) {
  return p.eval();
}

template <::sophus::concepts::ScalarType TPoint>
auto allTrue(TPoint const& p) -> bool {
  return bool(p);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto allTrue(TPoint const& p) -> bool {
  return p.all();
}

template <::sophus::concepts::ScalarType TPoint>
auto anyTrue(TPoint const& p) -> bool {
  return bool(p);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto anyTrue(TPoint const& p) -> bool {
  return p.any();
}

template <::sophus::concepts::ScalarType TPoint>
auto isFinite(TPoint const& p) -> bool {
  return std::isfinite(p);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto isFinite(TPoint const& p) -> bool {
  return p.array().isFinite().all();
}

template <::sophus::concepts::ScalarType TPoint>
auto isNan(TPoint const& p) -> bool {
  return std::isnan(p);
}

template <::sophus::concepts::PointType TPoint>
auto isNan(TPoint const& p) -> bool {
  return p.array().isNaN().all();
}

template <::sophus::concepts::ScalarType TPoint>
auto square(TPoint const& v) {
  return v * v;
}

template <::sophus::concepts::EigenDenseType TPoint>
auto square(TPoint const& v) {
  return v.squaredNorm();
}

template <::sophus::concepts::ScalarType TPoint>
auto min(TPoint const& a, TPoint const& b) -> TPoint {
  return std::min(a, b);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto min(TPoint const& a, TPoint const& b) -> TPoint {
  return a.cwiseMin(b);
}

template <::sophus::concepts::ScalarType TPoint>
auto max(TPoint const& a, TPoint const& b) -> TPoint {
  return std::max(a, b);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto max(TPoint const& a, TPoint const& b) -> TPoint {
  return a.cwiseMax(b);
}

template <::sophus::concepts::PointType TPoint>
auto clamp(TPoint const& val, TPoint const& lo, TPoint const& hi) -> TPoint {
  return sophus::max(lo, sophus::min(val, hi));
}

template <::sophus::concepts::ScalarType TPoint>
auto floor(TPoint s) {
  using std::floor;
  return floor(s);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto floor(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::floor(e);
  }
  return p;
}

template <::sophus::concepts::ScalarType TPoint>
auto ceil(TPoint s) {
  using std::ceil;
  return ceil(s);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto ceil(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::ceil(e);
  }
  return p;
}

template <::sophus::concepts::ScalarType TPoint>
auto round(TPoint s) {
  using std::ceil;
  return ceil(s);
}

template <::sophus::concepts::EigenDenseType TPoint>
auto round(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::round(e);
  }
  return p;
}

template <::sophus::concepts::ScalarType TPoint>
[[nodiscard]] auto plus(TPoint p, TPoint s) {
  p += s;
  return p;
}

template <::sophus::concepts::EigenDenseType TPoint>
[[nodiscard]] auto plus(TPoint p, typename TPoint::Scalar s) {
  p.array() += s;
  return p;
}

template <::sophus::concepts::ScalarType TPoint>
[[nodiscard]] auto isLessEqual(TPoint const& lhs, TPoint const& rhs) -> bool {
  return lhs <= rhs;
}

template <::sophus::concepts::EigenDenseType TPoint>
[[nodiscard]] auto isLessEqual(TPoint const& lhs, TPoint const& rhs) -> bool {
  return allTrue(eval(lhs.array() <= rhs.array()));
}

template <::sophus::concepts::ScalarType TPoint>
[[nodiscard]] auto tryGetElem(TPoint const& p, size_t row, size_t col = 0)
    -> Expected<TPoint> {
  if (row == 0 && col == 0) {
    return p;
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <::sophus::concepts::EigenDenseType TPoint>
[[nodiscard]] auto tryGetElem(TPoint const& p, size_t row, size_t col = 0)
    -> Expected<TPoint> {
  if (row < p.rows() && col < p.cols()) {
    return p(row, col);
  }
  return SOPHUS_UNEXPECTED(
      "({}, {}) access of array of size {} x {}", row, col, p.rows(), p.cols());
}

template <::sophus::concepts::ScalarType TPoint>
[[nodiscard]] auto trySetElem(TPoint& p, TPoint s, size_t row, size_t col = 0)
    -> Expected<Success> {
  if (row == 0 && col == 0) {
    p = s;
    return Success{};
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <::sophus::concepts::EigenDenseType TPoint>
[[nodiscard]] auto trySetElem(
    TPoint& p, typename TPoint::Scalar s, size_t row, size_t col = 0)
    -> Expected<Success> {
  if (row == 0 && col == 0) {
    p(row, col) = s;
    return Success{};
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <::sophus::concepts::ScalarType TPoint>
[[nodiscard]] auto transpose(TPoint p) {
  return p;
}

template <::sophus::concepts::EigenDenseType TPoint>
[[nodiscard]] auto transpose(TPoint p) {
  return p.transpose();
}

namespace details {

template <class TScalar, int kDim>
struct PointExamples;

template <class TScalar>
struct PointExamples<TScalar, 1> {
  static auto impl() {
    std::vector<Eigen::Vector<TScalar, 1>> point_vec;
    point_vec.push_back(Eigen::Vector<TScalar, 1>(TScalar(1)));
    point_vec.push_back(Eigen::Vector<TScalar, 1>(TScalar(-3)));
    point_vec.push_back(Eigen::Vector<TScalar, 1>::Zero());
    return point_vec;
  }
};

template <class TScalar>
struct PointExamples<TScalar, 2> {
  static auto impl() {
    std::vector<Eigen::Vector<TScalar, 2>> point_vec;
    point_vec.push_back(Eigen::Vector<TScalar, 2>(TScalar(1), TScalar(2)));
    point_vec.push_back(Eigen::Vector<TScalar, 2>(TScalar(1), TScalar(-3)));
    point_vec.push_back(Eigen::Vector<TScalar, 2>::Zero());
    point_vec.push_back(Eigen::Vector<TScalar, 2>::Ones());
    point_vec.push_back(Eigen::Vector<TScalar, 2>::UnitX());
    point_vec.push_back(Eigen::Vector<TScalar, 2>::UnitY());
    return point_vec;
  }
};

template <class TScalar>
struct PointExamples<TScalar, 3> {
  static auto impl() {
    std::vector<Eigen::Vector<TScalar, 3>> point_vec;
    point_vec.push_back(
        Eigen::Vector<TScalar, 3>(TScalar(1), TScalar(2), TScalar(0.1)));
    point_vec.push_back(
        Eigen::Vector<TScalar, 3>(TScalar(1), TScalar(-3), TScalar(-1)));
    point_vec.push_back(Eigen::Vector<TScalar, 3>::Zero());
    point_vec.push_back(Eigen::Vector<TScalar, 3>::Ones());
    point_vec.push_back(Eigen::Vector<TScalar, 3>::UnitX());
    point_vec.push_back(Eigen::Vector<TScalar, 3>::UnitZ());
    return point_vec;
  }
};

template <class TScalar>
struct PointExamples<TScalar, 4> {
  static auto impl() {
    std::vector<Eigen::Vector<TScalar, 4>> point_vec;
    point_vec.push_back(Eigen::Vector<TScalar, 4>(
        TScalar(1), TScalar(2), TScalar(0.1), TScalar(0.1)));
    point_vec.push_back(Eigen::Vector<TScalar, 4>(
        TScalar(1), TScalar(-3), TScalar(-1), TScalar(0.1)));
    point_vec.push_back(Eigen::Vector<TScalar, 4>::Zero());
    point_vec.push_back(Eigen::Vector<TScalar, 4>::Ones());
    point_vec.push_back(Eigen::Vector<TScalar, 4>::UnitX());
    point_vec.push_back(Eigen::Vector<TScalar, 4>::UnitZ());
    return point_vec;
  }
};

}  // namespace details

template <class TScalar, int kDim>
[[nodiscard]] auto pointExamples() {
  return details::PointExamples<TScalar, kDim>::impl();
}

}  // namespace sophus
