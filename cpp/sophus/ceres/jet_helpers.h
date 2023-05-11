// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

// Forward declare ceres::Jet, so we don't have to include <ceres/jet.h> here,
// and core does not have to depend on ceres.
namespace ceres {
template <class TScalar, int kN>
struct Jet;
}  // namespace ceres

namespace sophus {

namespace jet_helpers {

template <class TScalar>
struct GetValue {
  static auto impl(TScalar const& t) -> TScalar { return t; }
};

template <class TScalar, int kN>
struct GetValue<::ceres::Jet<TScalar, kN>> {
  static auto impl(TScalar const& t) -> TScalar { return t.a; }
};

}  // namespace jet_helpers
}  // namespace sophus
