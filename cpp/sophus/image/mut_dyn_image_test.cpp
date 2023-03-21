// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/dyn_image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

#define SOPHUS_TEST_IMG_EQ(lhs, rhs) /* NOLINT*/        \
  do {                                                  \
    SOPHUS_ASSERT_EQ(lhs.imageSize(), rhs.imageSize()); \
    for (int v = 0; v < lhs.imageSize().height; ++v) {  \
      for (int u = 0; u < lhs.imageSize().width; ++u) { \
        SOPHUS_ASSERT_EQ(lhs(u, v), rhs(u, v));         \
      }                                                 \
    }                                                   \
  } while (false)

TEST(AnyImage, create_access_and_extract) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  MutAnyImage<> mut_any_image(std::move(mut_image));
  SOPHUS_ASSERT(mut_image.isEmpty());

  SOPHUS_ASSERT_EQ(mut_any_image.imageSize(), size64);
  SOPHUS_ASSERT_EQ(mut_any_image.numChannels(), 1);
  SOPHUS_ASSERT_EQ(
      mut_any_image.pixelFormat().num_bytes_per_component, sizeof(float));
  SOPHUS_ASSERT_EQ(
      mut_any_image.pixelFormat().number_type, NumberType::floating_point);

  SOPHUS_ASSERT(!mut_any_image.has<uint16_t>());
  SOPHUS_ASSERT(!mut_any_image.has<double>());
  SOPHUS_ASSERT(mut_any_image.has<float>());
  SOPHUS_ASSERT(!(mut_any_image.has<Eigen::Vector3f>()));

  MutImage<float> mut_image2 = mut_any_image.moveOutAs<float>();
  SOPHUS_ASSERT(mut_any_image.isEmpty());
  SOPHUS_ASSERT(!mut_image2.isEmpty());
  SOPHUS_ASSERT_EQ(mut_image2.imageSize(), size64);

  mut_image2.mut(0, 0) = 0.9f;
  SOPHUS_ASSERT_EQ(mut_image2(0, 0), 0.9f);
}
