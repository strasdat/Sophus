// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/identity.h"

#include "sophus/concepts/group_accessors.h"
#include "sophus/concepts/lie_group_prop_tests.h"

#include <gtest/gtest.h>

namespace sophus::test {

TEST(identity, lie_group_prop_tests) {
  LieGroupPropTestSuite<Identity2<double>>::runAllTests("IdentityF64");
  LieGroupPropTestSuite<Identity3<float>>::runAllTests("IdentityF32");
}
}  // namespace sophus::test
