// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/spiral_similarity2.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(SpiralSimilarity2, unit_tests) {
  runSpiralSimilarity2UnitTests<SpiralSimilarity2<double>>();
  runSpiralSimilarity2UnitTests<SpiralSimilarity2<float>>();
}

TEST(SpiralSimilarity2, lie_group_prop_tests) {
  LieGroupPropTestSuite<SpiralSimilarity2<double>>::runAllTests(
      "SpiralSimilarity2F64");
  LieGroupPropTestSuite<SpiralSimilarity2<float>>::runAllTests(
      "SpiralSimilarity2F32");
}
}  // namespace sophus::test
