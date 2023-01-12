
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

template <PointType TPoint>
class Region;

using RegionI = Region<int>;
using RegionF32 = Region<float>;
using RegionF64 = Region<double>;

template <ScalarType TScalar>
using Region2 = Region<Eigen::Array<TScalar, 2, 1>>;
using Region2I = Region2<int>;
using Region2F32 = Region2<float>;
using Region2F64 = Region2<double>;

template <ScalarType TScalar>
using Region3 = Region<Eigen::Array<TScalar, 3, 1>>;
using Region3I = Region3<int>;
using Region3F32 = Region3<float>;
using Region3F64 = Region3<double>;

template <ScalarType TScalar>
using Region4 = Region<Eigen::Array<TScalar, 4, 1>>;
using Region4I = Region4<int>;
using Region4F32 = Region4<float>;
using Region4F64 = Region4<double>;

/// A region is a closed interval [a, b] with a being the lower bound (=min) and
/// b being the upper bound (=max).
///
/// Here, the bounds a, b sre either boths  scalars (e.g. floats, doubles)
/// or fixed length vectors/arrays (such as Eigen::Vector3f or Eigen::Array2d).
///
/// Special case for integer numbers:
///
///    An integer number X is considered being equivalent to a real region
///    [X-0.5, X+0.5].
template <PointType TPoint>
class Region {
 public:
  static bool constexpr kIsInteger = PointTraits<TPoint>::kIsInteger;
  using Point = TPoint;
  using Scalar = typename PointTraits<TPoint>::Scalar;

  static int constexpr kRows = PointTraits<TPoint>::kRows;
  static int constexpr kCols = PointTraits<TPoint>::kCols;
  static_assert(kRows >= 1);
  static_assert(kCols >= 1);
  static int constexpr kDim = kRows * kCols;

  /// Creates an uninitialized region.
  static Region<TPoint> uninitialized() noexcept { return Region(UninitTag{}); }

  /// Creates and empty region.
  static Region<TPoint> empty() noexcept {
    auto region = Region<TPoint>::uninitialized();
    region.min_max_ = {
        PointTraits<TPoint>::max(), PointTraits<TPoint>::lowest()};
    return region;
  }

  /// Creates unbounded region.
  ///
  /// If TPoint is floating point, the region is [-inf, +inf].
  static Region<TPoint> unbounded() noexcept {
    auto region = Region<TPoint>::uninitialized();
    region.min_max_ = {PointTraits<TPoint>::min(), PointTraits<TPoint>::max()};
    return region;
  }

  /// Creates a region from a given point.
  ///
  /// If TPoint is a floating point then the region is considered degenerated.
  static Region<TPoint> from(TPoint const& p) noexcept {
    auto region = Region<TPoint>::empty();
    SOPHUS_ASSERT(!isNan(p));
    region.extend(p);
    return region;
  }

  /// Creates Region from two points, min and max.
  ///
  /// The points min, max need not to be ordered. After construction it will be
  /// true that this->min() <= this->max().
  static Region<TPoint> fromMinMax(
      TPoint const& min, TPoint const& max) noexcept {
    auto region = Region<TPoint>::empty();
    SOPHUS_ASSERT(!isNan(min));
    SOPHUS_ASSERT(!isNan(max));
    region.extend(min);
    region.extend(max);
    return region;
  }

  /// Convenient constructor to create a  Segment from two points p1 and p2.
  ///
  /// The points need not to be ordered. After construction it will be true
  /// that this->min() <= this->max().
  ///
  /// Note: This constructor is only available for scalar regions. For
  /// multi-dim regions, use the  fromMinMax() factory instead.,
  template <class TScalar>
  requires std::is_same_v<TPoint, TScalar> Region(
      TScalar const& p1, TScalar const& p2)
  noexcept : min_max_{p1, p1} {
    SOPHUS_ASSERT(!isNan(p1));
    SOPHUS_ASSERT(!isNan(p2));
    this->extend(p2);
  }

  template <class TScalar>
  requires(kDim == 2) static Region2<TScalar> createPerAxis(
      Region<TScalar> const& segment_x,
      Region<TScalar> const& segment_y) noexcept {
    auto region = Region<TPoint>::uninitialized();
    region.setElem(segment_x, 0);
    region.setElem(segment_y, 1);
    return region;
  }

  template <class TScalar>
  requires(kDim == 3) static Region2<TScalar> createPerAxis(
      Region<TScalar> const& segment_x,
      Region<TScalar> const& segment_y,
      Region<TScalar> const& segment_z) noexcept {
    auto region = Region<TPoint>::uninitialized();
    region.setElem(segment_x, 0);
    region.setElem(segment_y, 1);
    region.setElem(segment_z, 2);
    return region;
  }

  template <class TScalar>
  requires(kDim == 4) static Region2<TScalar> createPerAxis(
      Region<TScalar> const& segment_x,
      Region<TScalar> const& segment_y,
      Region<TScalar> const& segment_z,
      Region<TScalar> const& segment_w) noexcept {
    auto region = Region<TPoint>::uninitialized();
    region.setElem(segment_x, 0);
    region.setElem(segment_y, 1);
    region.setElem(segment_z, 2);
    region.setElem(segment_w, 3);
    return region;
  }

  [[nodiscard]] Region<Scalar> const& getElem(size_t row) const {
    return {
        FARM_UNWRAP(tryGetElem(min(), row)),
        FARM_UNWRAP(tryGetElem(max(), row))};
  }

  void setElem(size_t row, Region<Scalar> const& s) {
    FARM_UNWRAP(trySetElem(min(), s, row));
    FARM_UNWRAP(trySetElem(max(), s, row));
  }

  /// Returns the lower bound of the region.
  [[nodiscard]] TPoint const& min() const noexcept { return min_max_[0]; }

  /// Returns the upper bound of the region.
  [[nodiscard]] TPoint const& max() const noexcept { return min_max_[1]; }

  /// Returns the clamped version of the given point.
  [[nodiscard]] TPoint clamp(TPoint const& point) const noexcept {
    return sophus::clamp(point, min_max_[0], min_max_[1]);
  }

  /// Returns true if the region contains the given point.
  [[nodiscard]] bool contains(TPoint const& point) const noexcept {
    return isLessEqual(min(), point) && isLessEqual(point, max());
  }

  /// Returns the range of the region.
  ///
  /// It is zero if the region is not proper.
  [[nodiscard]] TPoint range() const noexcept {
    if constexpr (kIsInteger) {
      //  For integers, we consider e.g. {2} == [1.5, 2.5] hence range of
      //  Region(2, 3) == [1.5, 3.5] is 2.
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

  /// Extends this by other region.
  Region<TPoint>& extend(Region const& other) noexcept {
    min_max_[0] = sophus::min(min(), other.min());
    min_max_[1] = sophus::max(max(), other.max());
    return *this;
  }

  /// Extends this by given point.
  Region<TPoint>& extend(TPoint const& point) noexcept {
    min_max_[0] = sophus::min(min(), point);
    min_max_[1] = sophus::max(max(), point);
    return *this;
  }

  /// Returns translated region.
  [[nodiscard]] Region<TPoint> translated(TPoint const& p) const noexcept {
    return Region<TPoint>::fromMinMax(min_max_[0] + p, min_max_[1] + p);
  }

  template <class TOtherPoint>
  Region<TOtherPoint> cast() const noexcept {
    if (isEmpty()) {
      return Region<TOtherPoint>::empty();
    }
    if constexpr (kIsInteger == Region<TOtherPoint>::kIsInteger) {
      // case 1: floating => floating  and integer => integer is trivial
      return Region<TOtherPoint>(
          sophus::cast<TOtherPoint>(min()), sophus::cast<TOtherPoint>(max()));
    }
    if constexpr (kIsInteger && !Region<TOtherPoint>::kIsInteger) {
      // case 2: integer to floating.
      //
      // example: [2, 5] -> [1.5, 5.5]
      return Region<TOtherPoint>(
          plus(sophus::cast<TOtherPoint>(min()), -0.5),
          plus(sophus::cast<TOtherPoint>(max()), 0.5));
    }
    // case 3: floating to integer.
    static_assert(
        kIsInteger || !Region<TOtherPoint>::kIsInteger,
        "For floating to integer: call encloseCast() or roundCast() "
        "instead.");
  }

  /// Returns the smallest integer region which contains this.
  ///
  /// example: [1.2, 1.3] -> [1, 2]
  template <class TOtherPoint>
  Region<TOtherPoint> encloseCast() const noexcept {
    static_assert(!kIsInteger && Region<TOtherPoint>::kIsInteger);
    return Region<TOtherPoint>(
        sophus::cast<TOtherPoint>(sophus::floor(min())),
        sophus::cast<TOtherPoint>(sophus::ceil(max())));
  }

  /// Rounds given region bounds and returns resulting integer region.
  ///
  /// example: [1.2, 2.3] -> [1, 2]
  /// example: [1.1, 2.7] -> [1, 3]
  template <class TOtherPoint>
  Region<TOtherPoint> roundCast() const noexcept {
    static_assert(!kIsInteger && Region<TOtherPoint>::kIsInteger);
    return Region<TOtherPoint>(
        sophus::cast<TOtherPoint>(sophus::round(min())),
        sophus::cast<TOtherPoint>(sophus::round(max())));
  }

  /// Returns true if region is empty.
  [[nodiscard]] bool isEmpty() const noexcept {
    return allTrue(this->min() == PointTraits<TPoint>::max()) ||
           allTrue(this->max() == PointTraits<TPoint>::lowest());
  }

  /// Returns true if region contains a single floating point number.
  [[nodiscard]] bool isDegenerated() const noexcept {
    return !kIsInteger && allTrue(min() == max());
  }

  /// Returns true if region is neither empty nor degenerated.
  /// Hence it contains a range of values.
  [[nodiscard]] bool isProper() const noexcept {
    return !this->isEmpty() && !this->isDegenerated();
  }

  /// Returns true if region has no bounds.
  [[nodiscard]] bool isUnbounded() const noexcept {
    return allTrue(this->min() == PointTraits<TPoint>::min()) &&
           allTrue(this->max() == PointTraits<TPoint>::max());
  }

 private:
  explicit Region(UninitTag) {}

  // invariant: this->isEmpty() or min_max[0] <= min_max[1]
  std::array<TPoint, 2> min_max_;
};

template <class TT>
bool operator==(Region<TT> const& lhs, Region<TT> const& rhs) {
  return lhs.min() == rhs.min() && lhs.max() == rhs.max();
}

}  // namespace sophus
