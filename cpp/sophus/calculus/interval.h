
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/common/point_methods.h"
#include "sophus/common/point_traits.h"

namespace sophus {

/// A closed interval [a, b] with a being the lower bound (=min) and b being the
/// upper bound (=max).
///
/// Here, the bounds a, b sre either boths  scalars (e.g. floats, doubles)
/// or fixed length vectors/arrays (such as Eigen::Vector3f or Eigen::Array2d).
///
/// Special case for integer numbers:
///
///    An integer number X is considered being equivalent to a real interval
///    [X-0.5, X+0.5].
template <PointType TPoint>
class Interval {
 public:
  static bool constexpr kIsInteger = PointTypeLimits<TPoint>::kIsInteger;
  using Point = TPoint;

  /// Creates an uninitialized interval.
  static Interval<TPoint> uninitialized() noexcept {
    return Interval(UninitTag{});
  }

  /// Creates and empty interval.
  static Interval<TPoint> empty() noexcept {
    auto interval = Interval<TPoint>::uninitialized();
    interval.min_max_ = {
        PointTypeLimits<TPoint>::max(), PointTypeLimits<TPoint>::lowest()};
    return interval;
  }

  /// Creates unbounded interval.
  ///
  /// If TPoint is floating point, the interval is [-inf, +inf].
  static Interval<TPoint> unbounded() noexcept {
    auto interval = Interval<TPoint>::uninitialized();
    interval.min_max_ = {
        PointTypeLimits<TPoint>::min(), PointTypeLimits<TPoint>::max()};
    return interval;
  }

  /// Creates an interval from a given point.
  ///
  /// If TPoint is a floating point then the interval is considered degenerated.
  static Interval<TPoint> from(TPoint const& p) noexcept {
    auto interval = Interval<TPoint>::empty();
    FARM_ASSERT(!isNan(p));
    interval.extend(p);
    return interval;
  }

  /// Creates Interval from two points, min and max.
  ///
  /// The points min, max need not to be ordered. After construction it will be
  /// true that this->min() <= this->max().
  static Interval<TPoint> fromMinMax(
      TPoint const& min, TPoint const& max) noexcept {
    auto interval = Interval<TPoint>::empty();
    FARM_ASSERT(!isNan(min));
    FARM_ASSERT(!isNan(max));
    interval.extend(min);
    interval.extend(max);
    return interval;
  }

  /// Convenient constructor to create a  Segment from two points p1 and p2.
  ///
  /// The points need not to be ordered. After construction it will be true
  /// that this->min() <= this->max().
  ///
  /// Note: This constructor is only available for scalar intervals (=
  /// segments). For multi-dim interval (= regions) use the  fromMinMax()
  /// factory instead.,
  template <class TScalar>
  requires std::is_same_v<TPoint, TScalar> Interval(
      TScalar const& p1, TScalar const& p2)
  noexcept : min_max_{p1, p1} {
    FARM_ASSERT(!isNan(p1));
    FARM_ASSERT(!isNan(p2));
    this->extend(p2);
  }

  /// Returns the lower bound of the interval.
  [[nodiscard]] TPoint const& min() const noexcept { return min_max_[0]; }

  /// Returns the upper bound of the interval.
  [[nodiscard]] TPoint const& max() const noexcept { return min_max_[1]; }

  /// Returns the clamped version of the given point.
  [[nodiscard]] TPoint clamp(TPoint const& point) const noexcept {
    return sophus::clamp(point, min_max_[0], min_max_[1]);
  }

  /// Returns true if the interval contains the given point.
  [[nodiscard]] bool contains(TPoint const& point) const noexcept {
    return isLessEqual(min(), point) && isLessEqual(point, max());
  }

  /// Returns the range of the interval.
  ///
  /// It is zero if the interval is not proper.
  [[nodiscard]] TPoint range() const noexcept {
    if constexpr (kIsInteger) {
      //  For integers, we consider e.g. {2} == [1.5, 2.5] hence range of
      //  Interval(2, 3) == [1.5, 3.5] is 2.
      return plus(eval(max() - min()), 1);
    }
    return eval(max() - min());
  }

  /// Returns the mid point.
  ///
  /// Note: If TPoint is an integer point then the result will be rounded to
  /// the closed integer.
  [[nodiscard]] TPoint mid() const noexcept {
    return eval(min() + range() / 2);
  }

  /// Extends this by other interval.
  Interval<TPoint>& extend(Interval const& other) noexcept {
    min_max_[0] = sophus::min(min(), other.min());
    min_max_[1] = sophus::max(max(), other.max());
    return *this;
  }

  /// Extends this by given point.
  Interval<TPoint>& extend(TPoint const& point) noexcept {
    min_max_[0] = sophus::min(min(), point);
    min_max_[1] = sophus::max(max(), point);
    return *this;
  }

  /// Returns translated interval.
  [[nodiscard]] Interval<TPoint> translated(TPoint const& p) const noexcept {
    return Interval<TPoint>::fromMinMax(min_max_[0] + p, min_max_[1] + p);
  }

  template <class TOtherPoint>
  Interval<TOtherPoint> cast() const noexcept {
    if (isEmpty()) {
      return Interval<TOtherPoint>::empty();
    }
    if constexpr (kIsInteger == Interval<TOtherPoint>::kIsInteger) {
      // case 1: floating => floating  and integer => integer is trivial
      return Interval<TOtherPoint>(
          sophus::cast<TOtherPoint>(min()), sophus::cast<TOtherPoint>(max()));
    }
    if constexpr (kIsInteger && !Interval<TOtherPoint>::kIsInteger) {
      // case 2: integer to floating.
      //
      // example: [2, 5] -> [1.5, 5.5]
      return Interval<TOtherPoint>(
          plus(sophus::cast<TOtherPoint>(min()), -0.5),
          plus(sophus::cast<TOtherPoint>(max()), 0.5));
    }
    // case 3: floating to integer.
    static_assert(
        kIsInteger || !Interval<TOtherPoint>::kIsInteger,
        "For floating to integer: call encloseCast() or roundCast() "
        "instead.");
  }

  /// Returns the smallest integer interval which contains this.
  ///
  /// example: [1.2, 1.3] -> [1, 2]
  template <class TOtherPoint>
  Interval<TOtherPoint> encloseCast() const noexcept {
    static_assert(!kIsInteger && Interval<TOtherPoint>::kIsInteger);
    return Interval<TOtherPoint>(
        sophus::cast<TOtherPoint>(sophus::floor(min())),
        sophus::cast<TOtherPoint>(sophus::ceil(max())));
  }

  /// Rounds given interval bounds and returns resulting integer interval.
  ///
  /// example: [1.2, 2.3] -> [1, 2]
  /// example: [1.1, 2.7] -> [1, 3]
  template <class TOtherPoint>
  Interval<TOtherPoint> roundCast() const noexcept {
    static_assert(!kIsInteger && Interval<TOtherPoint>::kIsInteger);
    return Interval<TOtherPoint>(
        sophus::cast<TOtherPoint>(sophus::round(min())),
        sophus::cast<TOtherPoint>(sophus::round(max())));
  }

  /// Returns true if interval is empty.
  [[nodiscard]] bool isEmpty() const noexcept {
    return allTrue(this->min() == PointTypeLimits<TPoint>::max()) ||
           allTrue(this->max() == PointTypeLimits<TPoint>::lowest());
  }

  /// Returns true if interval contains a single floating point number.
  [[nodiscard]] bool isDegenerated() const noexcept {
    return !kIsInteger && allTrue(min() == max());
  }

  /// Returns true if interval is neither empty nor degenerated.
  /// Hence it contains a range of values.
  [[nodiscard]] bool isProper() const noexcept {
    return !this->isEmpty() && !this->isDegenerated();
  }

  /// Returns true if interval has no bounds.
  [[nodiscard]] bool isUnbounded() const noexcept {
    return allTrue(this->min() == PointTypeLimits<TPoint>::min()) &&
           allTrue(this->max() == PointTypeLimits<TPoint>::max());
  }

 private:
  explicit Interval(UninitTag) {}

  // invariant: this->isEmpty() or min_max[0] <= min_max[1]
  std::array<TPoint, 2> min_max_;
};

template <class TT>
bool operator==(Interval<TT> const& lhs, Interval<TT> const& rhs) {
  return lhs.min() == rhs.min() && lhs.max() == rhs.max();
}

using SegmentI = Interval<int>;
using SegmentF32 = Interval<float>;
using SegmentF64 = Interval<double>;

using Region2I = Interval<Eigen::Vector2<int>>;
using Region2F32 = Interval<Eigen::Vector2<float>>;
using Region2F64 = Interval<Eigen::Vector2<double>>;

using Region3I = Interval<Eigen::Vector3<int>>;
using Region3F32 = Interval<Eigen::Vector3<float>>;
using Region3F64 = Interval<Eigen::Vector3<double>>;

using Region4I = Interval<Eigen::Vector4<int>>;
using Region4F32 = Interval<Eigen::Vector4<float>>;
using Region4F64 = Interval<Eigen::Vector4<double>>;

}  // namespace sophus
