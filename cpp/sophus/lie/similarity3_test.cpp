// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/similarity3.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(similarity3, unit_tests) {
  runSimilarity3UnitTests<Similarity3<double>>();
  runSimilarity3UnitTests<Similarity3<float>>();
}

TEST(identity, lie_group_prop_tests) {
  LieGroupPropTestSuite<Similarity3<double>>::runAllTests("Similarity3F64");
  LieGroupPropTestSuite<Similarity3<float>>::runAllTests("Similarity3F32");
}

}  // namespace sophus::test
