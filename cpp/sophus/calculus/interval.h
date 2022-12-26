
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/common/eigen_scalar_method.h"
#include "sophus/common/multi_dim_limits.h"

namespace sophus {

namespace {
template <class T>
concept Subtractable = requires(T a, T b) {
  b - a;
};
}  // namespace

// This works a lot like Eigens::AlignedBox, but supports
// TPixel as a scalar or vector element for use with generic
// image reductions etc
// The range of values is represented by min,max inclusive.
template <class TPixel>
class Interval {
 public:
  Interval() = default;
  Interval(Interval const&) = default;
  Interval(Interval&&) = default;
  Interval& operator=(Interval const&) = default;

  Interval(TPixel const& p) : min_max{p, p} {}

  Interval(TPixel const& p1, TPixel const& p2) : min_max{p1, p1} { extend(p2); }

  TPixel const& min() const { return min_max[0]; }
  TPixel const& max() const { return min_max[1]; }

  TPixel clamp(TPixel const& x) const {
    return sophus::max(sophus::min(x, min_max[1]), min_max[0]);
  }

  bool contains(TPixel const& x) const {
    return allTrue(eval(min() <= x)) && allTrue(eval(x <= max()));
  }

  auto fractionalPosition(Eigen::Array2d const& x) const {
    return eval(
        (x - sophus::cast<double>(min())) / sophus::cast<double>(range()));
  }

  // Only applicable if minmax object is valid()
  auto range() const requires Subtractable<TPixel> {
    return eval(max() - min());
  }
  auto mid() const { return eval(min() + range() / 2); }

  Interval<TPixel>& extend(Interval const& o) {
    min_max[0] = sophus::min(min(), o.min());
    min_max[1] = sophus::max(max(), o.max());
    return *this;
  }

  Interval<TPixel>& extend(TPixel const& p) {
    min_max[0] = sophus::min(min(), p);
    min_max[1] = sophus::max(max(), p);
    return *this;
  }

  Interval<TPixel> translated(TPixel const& p) const {
    return {min_max[0] + p, min_max[1] + p};
  }

  template <class To>
  Interval<To> cast() const {
    return Interval<To>(sophus::cast<To>(min()), sophus::cast<To>(max()));
  }

  bool empty() const {
    return min_max[0] == MultiDimLimits<TPixel>::max() ||
           min_max[1] == MultiDimLimits<TPixel>::lowest();
  }

  static Interval<TPixel> open() {
    Interval<TPixel> mm;
    mm.min_max = {MultiDimLimits<TPixel>::min(), MultiDimLimits<TPixel>::max()};
    return mm;
  }

  static Interval<TPixel> closed() {
    Interval<TPixel> mm;
    mm.min_max = {MultiDimLimits<TPixel>::max(), MultiDimLimits<TPixel>::min()};
    return mm;
  }

 private:
  // invariant that min_max[0] <= min_max[1]
  // or min_max is as below when uninitialized
  std::array<TPixel, 2> min_max = {
      MultiDimLimits<TPixel>::max(), MultiDimLimits<TPixel>::min()};
};

namespace details {
template <class TScalar>
class Cast<Interval<TScalar>> {
 public:
  template <class To>
  static Interval<To> impl(Interval<TScalar> const& v) {
    return v.template cast<To>();
  }
  template <class To>
  static auto implScalar(Interval<TScalar> const& v) {
    using ElT = decltype(cast<To>(std::declval<TScalar>()));
    return v.template cast<ElT>();
  }
};
}  // namespace details

template <class T>
bool operator==(Interval<T> const& lhs, Interval<T> const& rhs) {
  return lhs.min() == rhs.min() && lhs.max() == rhs.max();
}

template <class T>
auto relative(T p, Interval<T> region) {
  return (p - region.min()).eval();
}

template <class T>
auto normalized(T p, Interval<T> region) {
  return (cast<double>(p - region.min()) / cast<double>(region.range())).eval();
}

}  // namespace sophus
