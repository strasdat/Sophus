// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/rotation3.h"

namespace sophus {

template <class TScalar>
using SO3 = Rotation3<TScalar>;  // NOLINT
using SO3f = Rotation3<float>;   // NOLINT
using SO3d = Rotation3<double>;  // NOLINT

}  // namespace sophus
