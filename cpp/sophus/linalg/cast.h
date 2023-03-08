
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

namespace details {

template <class TScalar>
class Cast {
 public:
  template <class TTo>
  static auto impl(TScalar const& s) -> TTo {
    return static_cast<TTo>(s);
  }
  template <class TTo>
  static auto implScalar(TScalar const& s) -> TTo {
    return static_cast<TTo>(s);
  }
};

template <::sophus::concepts::EigenType TT>
class Cast<TT> {
 public:
  template <class TTo>
  static auto impl(TT const& v) {
    return v.template cast<typename TTo::Scalar>().eval();
  }
  template <class TTo>
  static auto implScalar(TT const& v) {
    return v.template cast<TTo>().eval();
  }
};

template <class TT>
class Cast<std::vector<TT>> {
 public:
  template <class TTo>
  static auto impl(std::vector<TT> const& v) {
    using ToEl = std::decay_t<decltype(*std::declval<TTo>().data())>;
    std::vector<ToEl> r;
    r.reserve(v.size());
    for (auto const& el : v) {
      r.push_back(Cast<TT>::template impl<ToEl>(el));
    }
    return r;
  }
  template <class TTo>
  static auto implScalar(std::vector<TT> const& v) {
    using ToEl = decltype(Cast<TT>::template implScalar<TTo>(v[0]));
    std::vector<ToEl> r;
    r.reserve(v.size());
    for (auto const& el : v) {
      r.push_back(Cast<TT>::template impl<ToEl>(el));
    }
    return r;
  }
};

}  // namespace details

template <class TTo, class TT>
auto cast(const TT& p) {
  return details::Cast<TT>::template impl<TTo>(p);
}

template <::sophus::concepts::Arithmetic TTo, class TT>
auto cast(const TT& p) {
  return details::Cast<TT>::template implScalar<TTo>(p);
}

}  // namespace sophus
