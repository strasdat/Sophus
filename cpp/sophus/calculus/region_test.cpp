// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/calculus/region.h"

#include <gtest/gtest.h>

using namespace sophus;

template <concepts::PointType TPoint>
void regionTests(std::vector<TPoint> const& points) {
  // TODO: test fromMinMax, createPerAxis, cast, encloseCast, roundCast, clamp,
  // extend

  using RegionP = Region<TPoint>;
  std::vector<RegionP> regions;

  // constr
  RegionP empty_region = RegionP::empty();
  regions.push_back(empty_region);
  EXPECT_TRUE(empty_region.isEmpty());
  EXPECT_FALSE(empty_region.isDegenerated());
  EXPECT_FALSE(empty_region.isProper());
  EXPECT_FALSE(empty_region.isUnbounded());
  for (TPoint p : points) {
    EXPECT_FALSE(empty_region.contains(p));
  }

  RegionP unbounded = RegionP::unbounded();
  regions.push_back(unbounded);

  EXPECT_FALSE(unbounded.isEmpty());
  EXPECT_FALSE(unbounded.isDegenerated());
  EXPECT_TRUE(unbounded.isProper());
  EXPECT_TRUE(unbounded.isUnbounded());
  for (TPoint p : points) {
    EXPECT_TRUE(unbounded.contains(p));
  }

  for (TPoint p : points) {
    auto x = RegionP::from(p);
    regions.push_back(x);
    EXPECT_FALSE(x.isEmpty());
    EXPECT_TRUE(x.contains(p));
    if (RegionP::kIsInteger) {
      EXPECT_TRUE(x.isProper());
      EXPECT_FALSE(x.isDegenerated());
    } else {
      EXPECT_FALSE(x.isProper());
      EXPECT_TRUE(x.isDegenerated());
    }
    EXPECT_FALSE(x.isUnbounded());
  }

  for (int i = 0; i < RegionP::kDim; ++i) {
    for (auto x : regions) {
      if (x.isEmpty()) {
        EXPECT_FALSE(x.tryMin());
        EXPECT_FALSE(x.tryMax());
        EXPECT_EQ(x.range(), zero<TPoint>());
      } else {
        auto maybe_min = x.tryMin();
        auto maybe_max = x.tryMax();
        EXPECT_EQ(SOPHUS_UNWRAP(maybe_min), x.min());
        EXPECT_EQ(SOPHUS_UNWRAP(maybe_max), x.max());
      }
    }
  }
}

TEST(ScalarRegion, init) {
  regionTests<float>({0.0, 1.0, -5.0});
  regionTests<uint8_t>({0u, 1u, 128u, 255u});
  regionTests<Eigen::Vector2d>(pointExamples<double, 2>());
  regionTests<Eigen::Vector3i>(pointExamples<int, 3>());
  regionTests<Eigen::Vector4f>(pointExamples<float, 4>());
}
