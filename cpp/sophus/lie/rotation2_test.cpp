// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/rotation2.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(rotation2, unit_tests) {
  runRotation2UnitTests<Rotation2<double>>();
  runRotation2UnitTests<Rotation2<float>>();
}

TEST(rotation2, lie_group_prop_tests) {
  LieGroupPropTestSuite<Rotation2<double>>::runAllTests("Rotation2F64");
  LieGroupPropTestSuite<Rotation2<float>>::runAllTests("Rotation2F32");
}

}  // namespace sophus::test
