// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/calculus/interval.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(Interval, unit) {
  // constr
  auto empty_f64 = SegmentF64::empty();
  EXPECT_TRUE(empty_f64.isEmpty());
  EXPECT_FALSE(empty_f64.isDegenerated());
  EXPECT_FALSE(empty_f64.isProper());
  EXPECT_FALSE(empty_f64.isUnbounded());

  auto unbounded_f32 = SegmentF32::unbounded();
  EXPECT_FALSE(unbounded_f32.isEmpty());
  EXPECT_FALSE(unbounded_f32.isDegenerated());
  EXPECT_TRUE(unbounded_f32.isProper());
  EXPECT_TRUE(unbounded_f32.isUnbounded());

  auto one_i = SegmentI::from(1);
  EXPECT_FALSE(one_i.isEmpty());
  EXPECT_FALSE(one_i.isDegenerated());
  EXPECT_TRUE(one_i.isProper());
  EXPECT_FALSE(one_i.isUnbounded());

  auto two_f32 = SegmentF32::from(2.f);
  EXPECT_FALSE(two_f32.isEmpty());
  EXPECT_TRUE(two_f32.isDegenerated());
  EXPECT_FALSE(two_f32.isProper());
  EXPECT_FALSE(two_f32.isUnbounded());
}
