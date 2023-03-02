// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/rotation2.h"
#include "sophus/lie/rotation3.h"
#include "sophus/lie/scaling.h"
#include "sophus/lie/similarity2.h"
#include "sophus/lie/similarity3.h"
#include "sophus/lie/spiral_similarity2.h"
#include "sophus/lie/spiral_similarity3.h"
#include "sophus/lie/translation.h"

namespace sophus {
namespace interp_details {

template <class TGroup>
struct Traits;

template <class TScalar, int kPointDim>
struct Traits<Translation<TScalar, kPointDim>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(
      Translation<TScalar, kPointDim> const& /*unused*/) -> bool {
    return false;
  }
};

template <class TScalar>
struct Traits<Rotation2<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(Rotation2<TScalar> const& foo_from_bar)
      -> bool {
    using std::abs;
    TScalar angle = abs(foo_from_bar.log()[0]);
    TScalar const k_pi = kPi<TScalar>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<TScalar>;
  }
};

template <class TScalar>
struct Traits<SpiralSimilarity2<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(
      SpiralSimilarity2<TScalar> const& foo_from_bar) -> bool {
    return Traits<Rotation2<TScalar>>::hasShortestPathAmbiguity(
        foo_from_bar.rotation());
  }
};

template <class TScalar>
struct Traits<Rotation3<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(Rotation3<TScalar> const& foo_from_bar)
      -> bool {
    using std::abs;
    TScalar angle = abs(foo_from_bar.log().norm());
    TScalar const k_pi = kPi<TScalar>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<TScalar>;
  }
};

template <class TScalar>
struct Traits<SpiralSimilarity3<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(
      SpiralSimilarity3<TScalar> const& foo_from_bar) -> bool {
    return Traits<Rotation3<TScalar>>::hasShortestPathAmbiguity(
        foo_from_bar.rotation());
  }
};

template <class TScalar>
struct Traits<Isometry2<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(Isometry2<TScalar> const& foo_from_bar)
      -> bool {
    return Traits<Rotation2<TScalar>>::hasShortestPathAmbiguity(
        foo_from_bar.rotation());
  }
};

template <class TScalar>
struct Traits<Isometry3<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(Isometry3<TScalar> const& foo_from_bar)
      -> bool {
    return Traits<Rotation3<TScalar>>::hasShortestPathAmbiguity(
        foo_from_bar.so3());
  }
};

template <class TScalar>
struct Traits<Similarity2<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(Similarity2<TScalar> const& foo_from_bar)
      -> bool {
    return Traits<Rotation2<TScalar>>::hasShortestPathAmbiguity(
        foo_from_bar.spiralSimilarity().rotation());
    ;
  }
};

template <class TScalar>
struct Traits<Similarity3<TScalar>> {
  static bool constexpr kSupported = true;

  static auto hasShortestPathAmbiguity(Similarity3<TScalar> const& foo_from_bar)
      -> bool {
    return Traits<Rotation3<TScalar>>::hasShortestPathAmbiguity(
        foo_from_bar.spiralSimilarity().rotation());
    ;
  }
};

}  // namespace interp_details
}  // namespace sophus
