// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/roto_scaling3.h"

#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(Scaling, lie_group_prop_tests) {
  LieGroupPropTestSuite<RotoScaling3<double>>::runAllTests("RotoScaling3F64");
  LieGroupPropTestSuite<RotoScaling3<float>>::runAllTests("RotoScaling3F32");
}
}  // namespace sophus::test
