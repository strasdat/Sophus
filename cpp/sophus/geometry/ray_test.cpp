// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/ray.h"

#include <farm_ng/core/logging/eigen.h>
#include <farm_ng/core/logging/logger.h>
#include <gtest/gtest.h>

using namespace sophus;

TEST(unitvec, unit2) {
  EXPECT_NO_FATAL_FAILURE(UnitVector2<double>::fromUnitVector({1.0, 0.0}));
  EXPECT_NO_FATAL_FAILURE(UnitVector2<double>::fromUnitVector({0.0, 1.0}));
  EXPECT_FALSE(UnitVector2<double>::tryFromUnitVector({0.0, 0.5}));
}

TEST(unitvec, unit3) {
  EXPECT_NO_FATAL_FAILURE(UnitVector3<double>::fromUnitVector({1.0, 0.0, 0.0}));
  EXPECT_NO_FATAL_FAILURE(UnitVector3<double>::fromUnitVector({0.0, 1.0, 0.0}));
  EXPECT_NO_FATAL_FAILURE(UnitVector3<double>::fromUnitVector({0.0, 0.0, 1.0}));
  EXPECT_FALSE(UnitVector3<double>::tryFromUnitVector({0.0, 0.5, 0.0}));
}

TEST(unitvec, copy3) {
  auto a = UnitVector3<double>::fromUnitVector({1.0, 0.0, 0.0});
  auto b = UnitVector3<double>::fromUnitVector({0.0, 1.0, 0.0});
  UnitVector3<double> c = a;
  FARM_CHECK_NEAR(a.vector(), c.vector(), kEpsilonF64);

  c = UnitVector3<double>(b);
  FARM_CHECK_NEAR(b.vector(), c.vector(), kEpsilonF64);

  auto d = (a = b);
  FARM_CHECK_NEAR(d.vector(), a.vector(), kEpsilonF64);
  FARM_CHECK_NEAR(d.vector(), b.vector(), kEpsilonF64);
}
