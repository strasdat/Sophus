// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/rotation3.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(rotation3, unit_tests) {
  runRotation3UnitTests<Rotation3<double>>();
  runRotation3UnitTests<Rotation3<float>>();
}

TEST(rotation3, lie_group_prop_tests) {
  LieGroupPropTestSuite<Rotation3<double>>::runAllTests("Rotation3F64");
  LieGroupPropTestSuite<Rotation3<float>>::runAllTests("Rotation3F32");
}
}  // namespace sophus::test
