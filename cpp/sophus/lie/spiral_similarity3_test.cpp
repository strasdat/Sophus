// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/spiral_similarity3.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(SpiralSimilarity3_lie_group_prop_tests_Test, unit_tests) {
  runSpiralSimilarity3UnitTests<SpiralSimilarity3<double>>();
  runSpiralSimilarity3UnitTests<SpiralSimilarity3<float>>();
}

TEST(SpiralSimilarity3, lie_group_prop_tests) {
  LieGroupPropTestSuite<SpiralSimilarity3<double>>::runAllTests(
      "SpiralSimilarity3F64");
  LieGroupPropTestSuite<SpiralSimilarity3<float>>::runAllTests(
      "SpiralSimilarity3F32");
}
}  // namespace sophus::test
