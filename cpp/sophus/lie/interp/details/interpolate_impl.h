// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/cartesian.h"
#include "sophus/lie/rxso2.h"
#include "sophus/lie/rxso3.h"
#include "sophus/lie/se2.h"
#include "sophus/lie/se3.h"
#include "sophus/lie/sim2.h"
#include "sophus/lie/sim3.h"
#include "sophus/lie/so2.h"
#include "sophus/lie/so3.h"

namespace sophus {
namespace interp_details {

template <class TGroup>
struct Traits;

template <class TScalar, int kPointDim>
struct Traits<Cartesian<TScalar, kPointDim>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(
      Cartesian<TScalar, kPointDim> const& /*unused*/) {
    return false;
  }
};

template <class TScalar>
struct Traits<So2<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(So2<TScalar> const& foo_transform_bar) {
    using std::abs;
    TScalar angle = abs(foo_transform_bar.log());
    TScalar const k_pi = kPi<TScalar>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<TScalar>;
  }
};

template <class TScalar>
struct Traits<RxSo2<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(
      RxSo2<TScalar> const& foo_transform_bar) {
    return Traits<So2<TScalar>>::hasShortestPathAmbiguity(
        foo_transform_bar.so2());
  }
};

template <class TScalar>
struct Traits<So3<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(So3<TScalar> const& foo_transform_bar) {
    using std::abs;
    TScalar angle = abs(foo_transform_bar.logAndTheta().theta);
    TScalar const k_pi = kPi<TScalar>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<TScalar>;
  }
};

template <class TScalar>
struct Traits<RxSo3<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(
      RxSo3<TScalar> const& foo_transform_bar) {
    return Traits<So3<TScalar>>::hasShortestPathAmbiguity(
        foo_transform_bar.so3());
  }
};

template <class TScalar>
struct Traits<Se2<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Se2<TScalar> const& foo_transform_bar) {
    return Traits<So2<TScalar>>::hasShortestPathAmbiguity(
        foo_transform_bar.so2());
  }
};

template <class TScalar>
struct Traits<Se3<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Se3<TScalar> const& foo_transform_bar) {
    return Traits<So3<TScalar>>::hasShortestPathAmbiguity(
        foo_transform_bar.so3());
  }
};

template <class TScalar>
struct Traits<Sim2<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Sim2<TScalar> const& foo_transform_bar) {
    return Traits<So2<TScalar>>::hasShortestPathAmbiguity(
        foo_transform_bar.rxso2().so2());
    ;
  }
};

template <class TScalar>
struct Traits<Sim3<TScalar>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Sim3<TScalar> const& foo_transform_bar) {
    return Traits<So3<TScalar>>::hasShortestPathAmbiguity(
        foo_transform_bar.rxso3().so3());
    ;
  }
};

}  // namespace interp_details
}  // namespace sophus
