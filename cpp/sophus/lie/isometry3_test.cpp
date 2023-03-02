// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/isometry3.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(isometry3, unit_tests) {
  runIsometry3UnitTests<Isometry3<double>>();
  runIsometry3UnitTests<Isometry3<float>>();
}

TEST(Isometry, lie_group_prop_tests) {
  LieGroupPropTestSuite<Isometry3<double>>::runAllTests("Isometry3F64");
  LieGroupPropTestSuite<Isometry3<float>>::runAllTests("Isometry3F32");
}

}  // namespace sophus::test
