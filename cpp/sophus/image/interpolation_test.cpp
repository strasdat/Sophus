// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/interpolation.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(interpolation, unit) {
  int w = 3;
  int h = 2;

  sophus::MutImage<float> img({3, 2});
  img.fill(0.f);

  for (float v = 0; v <= h - 1.f; v += 0.1) {
    for (float u = 0; u <= w - 1.f; u += 0.1) {
      float val = interpolate(img, Eigen::Vector2f(u, v));
      SOPHUS_ASSERT_EQ(val, 0.f);
    }
  }

  int u = 1;
  int v = 0;
  img.mut(u, v) = 0.5f;

  float val = interpolate(img, Eigen::Vector2f(0, 0));
  SOPHUS_ASSERT_EQ(val, 0.f);

  val = interpolate(img, Eigen::Vector2f(1, 0));
  SOPHUS_ASSERT_EQ(val, 0.5f);

  val = interpolate(img, Eigen::Vector2f(0.25, 0));
  SOPHUS_ASSERT_EQ(val, 0.125f);

  val = interpolate(img, Eigen::Vector2f(0.5, 0));
  SOPHUS_ASSERT_EQ(val, 0.25f);

  val = interpolate(img, Eigen::Vector2f(0.75, 0));
  SOPHUS_ASSERT_EQ(val, 0.375f);

  u = 1;
  v = 1;
  img.mut(u, v) = 1.f;

  val = interpolate(img, Eigen::Vector2f(0, 0));
  SOPHUS_ASSERT_EQ(val, 0.f);

  val = interpolate(img, Eigen::Vector2f(1.0, 0.5));
  SOPHUS_ASSERT_EQ(val, 0.75f);

  val = interpolate(img, Eigen::Vector2f(0.5, 0.5));
  SOPHUS_ASSERT_EQ(val, 0.375f);
}
