// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/translation.h"

#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(translation, lie_group_prop_tests) {
  LieGroupPropTestSuite<Translation<double, 2>>::runAllTests("Translation2F64");
  LieGroupPropTestSuite<Translation<float, 2>>::runAllTests("Translation2F32");
  LieGroupPropTestSuite<Translation<double, 3>>::runAllTests("Translation3F64");
  LieGroupPropTestSuite<Translation<float, 3>>::runAllTests("Translation3F32");
}
}  // namespace sophus::test
