// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/similarity2.h"

#include "sophus/concepts/group_accessors_unit_tests.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(similarity2, unit_tests) {
  runSimilarity2UnitTests<Similarity2<double>>();
  runSimilarity2UnitTests<Similarity2<float>>();
}

TEST(Similarity2, lie_group_prop_tests) {
  LieGroupPropTestSuite<Similarity2<double>>::runAllTests("Similarity2F64");
  LieGroupPropTestSuite<Similarity2<float>>::runAllTests("Similarity2F32");
}

}  // namespace sophus::test
