// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/quaternion.h"

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::Quaternion<double>>;
template class Map<sophus::Quaternion<double> const>;
}  // namespace Eigen

namespace sophus {

template class Quaternion<double>;
#if SOPHUS_CERES
template class Quaternion<ceres::Jet<double, 3>>;
#endif
int main() { return 0; }
