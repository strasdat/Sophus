
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

#include <algorithm>
#include <utility>
#include <vector>

namespace sophus {
namespace concepts {

namespace details {

template <class TScalar>
class Reduce {
 public:
  using Aggregate = TScalar;

  template <class TReduce, class TFunc>
  static void implUnary(TScalar const& s, TReduce& reduce, TFunc const& f) {
    f(s, reduce);
  }

  template <class TReduce, class TFunc>
  static void implBinary(
      TScalar const& a, TScalar const& b, TReduce& reduce, TFunc const& f) {
    f(a, b, reduce);
  }
};

template <::sophus::concepts::EigenDenseType TT>
class Reduce<TT> {
 public:
  template <class TReduce, class TFunc>
  static void implUnary(TT const& v, TReduce& reduce, TFunc const& f) {
    for (int r = 0; r < v.rows(); ++r) {
      for (int c = 0; c < v.cols(); ++c) {
        f(v(r, c), reduce);
      }
    }
  }

  template <class TReduce, class TFunc>
  static void implBinary(
      TT const& a, TT const& b, TReduce& reduce, TFunc const& f) {
    for (int r = 0; r < a.rows(); ++r) {
      for (int c = 0; c < a.cols(); ++c) {
        f(a(r, c), b(r, c), reduce);
      }
    }
  }
};

}  // namespace details

template <class TPoint, class TFunc, class TReduce>
void reduceArg(TPoint const& x, TReduce& reduce, TFunc&& func) {
  details::Reduce<TPoint>::impl_unary(x, reduce, std::forward<TFunc>(func));
}

template <class TPoint, class TFunc, class TReduce>
void reduceArg(
    TPoint const& a, TPoint const& b, TReduce& reduce, TFunc&& func) {
  details::Reduce<TPoint>::impl_binary(a, b, reduce, std::forward<TFunc>(func));
}

template <class TPoint, class TFunc, class TReduce>
auto reduce(TPoint const& x, TReduce const& initial, TFunc&& func) -> TReduce {
  TReduce reduce = initial;
  details::Reduce<TPoint>::impl_unary(x, reduce, std::forward<TFunc>(func));
  return reduce;
}

template <class TPoint, class TFunc, class TReduce>
auto reduce(
    TPoint const& a, TPoint const& b, TReduce const& initial, TFunc&& func)
    -> TReduce {
  TReduce reduce = initial;
  details::Reduce<TPoint>::impl_binary(a, b, reduce, std::forward<TFunc>(func));
  return reduce;
}

}  // namespace concepts
}  // namespace sophus
