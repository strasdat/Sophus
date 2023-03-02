// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/isometry2.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

using namespace sophus;

namespace sophus::test {

TEST(isometry2, unit_tests) {
  runIsometry2UnitTests<Isometry2<double>>();
  runIsometry2UnitTests<Isometry2<float>>();
}

TEST(identity, lie_group_prop_tests) {
  LieGroupPropTestSuite<Isometry2<double>>::runAllTests("Isometry2F64");
  LieGroupPropTestSuite<Isometry2<float>>::runAllTests("Isometry2F32");
}

}  // namespace sophus::test
