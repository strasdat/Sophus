// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/isometry3.h"

namespace sophus {

template <class TScalar>
using SE3 = Isometry3<TScalar>;  // NOLINT
using SE3f = Isometry3<float>;   // NOLINT
using SE3d = Isometry3<double>;  // NOLINT

}  // namespace sophus
