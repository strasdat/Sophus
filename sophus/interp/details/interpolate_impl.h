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

template <class GroupT>
struct Traits;

template <class ScalarT, int kPointDim>
struct Traits<Cartesian<ScalarT, kPointDim>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(
      Cartesian<ScalarT, kPointDim> const& /*unused*/) {
    return false;
  }
};

template <class ScalarT>
struct Traits<So2<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(So2<ScalarT> const& foo_transform_bar) {
    using std::abs;
    ScalarT angle = abs(foo_transform_bar.log());
    ScalarT const k_pi = kPi<ScalarT>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<ScalarT>;
  }
};

template <class ScalarT>
struct Traits<RxSo2<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(
      RxSo2<ScalarT> const& foo_transform_bar) {
    return Traits<So2<ScalarT>>::hasShortestPathAmbiguity(
        foo_transform_bar.so2());
  }
};

template <class ScalarT>
struct Traits<So3<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(So3<ScalarT> const& foo_transform_bar) {
    using std::abs;
    ScalarT angle = abs(foo_transform_bar.logAndTheta().theta);
    ScalarT const k_pi = kPi<ScalarT>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<ScalarT>;
  }
};

template <class ScalarT>
struct Traits<RxSo3<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(
      RxSo3<ScalarT> const& foo_transform_bar) {
    return Traits<So3<ScalarT>>::hasShortestPathAmbiguity(
        foo_transform_bar.so3());
  }
};

template <class ScalarT>
struct Traits<Se2<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Se2<ScalarT> const& foo_transform_bar) {
    return Traits<So2<ScalarT>>::hasShortestPathAmbiguity(
        foo_transform_bar.so2());
  }
};

template <class ScalarT>
struct Traits<Se3<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Se3<ScalarT> const& foo_transform_bar) {
    return Traits<So3<ScalarT>>::hasShortestPathAmbiguity(
        foo_transform_bar.so3());
  }
};

template <class ScalarT>
struct Traits<Sim2<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Sim2<ScalarT> const& foo_transform_bar) {
    return Traits<So2<ScalarT>>::hasShortestPathAmbiguity(
        foo_transform_bar.rxso2().so2());
    ;
  }
};

template <class ScalarT>
struct Traits<Sim3<ScalarT>> {
  static bool constexpr kSupported = true;

  static bool hasShortestPathAmbiguity(Sim3<ScalarT> const& foo_transform_bar) {
    return Traits<So3<ScalarT>>::hasShortestPathAmbiguity(
        foo_transform_bar.rxso3().so3());
    ;
  }
};

}  // namespace interp_details
}  // namespace sophus
