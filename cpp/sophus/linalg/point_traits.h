// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/point.h"

#include <Eigen/Core>

#include <limits>

namespace sophus {

template <class TPoint>
struct PointTraits;

template <concepts::ScalarType TPoint>
struct PointTraits<TPoint> {
  using Scalar = TPoint;

  static bool constexpr kIsFloatingPoint = std::is_floating_point_v<Scalar>;
  static bool constexpr kIsInteger = std::is_integral_v<Scalar>;

  static int constexpr kRows = 1;
  static int constexpr kCols = 1;

  static bool constexpr kHasInfinity =
      std::numeric_limits<Scalar>::has_infinity;
  static bool constexpr kHasQuietNan =
      std::numeric_limits<Scalar>::has_quiet_NaN;
  static bool constexpr kHasSignalingNan =
      std::numeric_limits<Scalar>::has_signaling_NaN;

  static TPoint lowest() { return std::numeric_limits<Scalar>::lowest(); };
  static TPoint min() { return std::numeric_limits<Scalar>::min(); };
  static TPoint max() { return std::numeric_limits<Scalar>::max(); };
};

template <concepts::EigenDenseType TPoint>
struct PointTraits<TPoint> {
  using Scalar = typename TPoint::Scalar;
  static int constexpr kRows = TPoint::RowsAtCompileTime;
  static int constexpr kCols = TPoint::ColsAtCompileTime;

  static bool constexpr kIsFloatingPoint = std::is_floating_point_v<Scalar>;
  static bool constexpr kIsInteger = std::is_integral_v<Scalar>;

  static bool constexpr kHasInfinity =
      std::numeric_limits<Scalar>::has_infinity;
  static bool constexpr kHasQuietNan =
      std::numeric_limits<Scalar>::has_quiet_NaN;
  static bool constexpr kHasSignalingNan =
      std::numeric_limits<Scalar>::has_signaling_NaN;

  static TPoint lowest() {
    return TPoint::Constant(std::numeric_limits<Scalar>::lowest());
  };
  static TPoint min() {
    return TPoint::Constant(std::numeric_limits<Scalar>::min());
  };
  static TPoint max() {
    return TPoint::Constant(std::numeric_limits<Scalar>::max());
  };
  static TPoint epsilon() {
    return TPoint::Constant(std::numeric_limits<Scalar>::epsilon());
  };
  /// ... plus a bunch more if we need them
};

}  // namespace sophus
