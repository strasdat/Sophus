// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(ImageBool, logic) {
  // clang-format off
  const std::array<uint16_t, 6> data_u16 =
      { 0,  1,  2,
       10, 11, 12};
  // clang-format on

  ImageSize image_size(3, 2);
  ImageView<uint16_t> view(image_size, data_u16.data());
  ImageView<uint16_t> view2(image_size, data_u16.data());

  MutImageBool equal_mask = isEqualMask(view, view2);

  SOPHUS_ASSERT_EQ(countTrue(equal_mask), 6);
  SOPHUS_ASSERT(isAllTrue(equal_mask));
  SOPHUS_ASSERT(isAnyTrue(equal_mask));

  equal_mask.mut(1, 1) = false;
  SOPHUS_ASSERT_EQ(countTrue(equal_mask), 5);
  SOPHUS_ASSERT(!isAllTrue(equal_mask));
  SOPHUS_ASSERT(isAnyTrue(equal_mask));
  auto maybe_pixel = firstFalsePixel(equal_mask);
  Eigen::Vector2i first_false_pixel = SOPHUS_UNWRAP(maybe_pixel);
  SOPHUS_ASSERT_EQ(first_false_pixel, Eigen::Vector2i(1, 1));
  auto maybe_true_pixel = firstTruePixel(equal_mask);
  Eigen::Vector2i first_true_pixel = SOPHUS_UNWRAP(maybe_true_pixel);
  SOPHUS_ASSERT_EQ(first_true_pixel, Eigen::Vector2i(0, 0));
}

TEST(SOPHUS_ASSERT_IMAGE_EQ, death_test) {
  MutImage<float> mut_image({2, 3});
  mut_image.fill(0.25f);

  MutImage<float> mut_image2 = MutImage<float>::makeCopyFrom(mut_image);

  SOPHUS_ASSERT_IMAGE_EQ(mut_image, mut_image2);

  mut_image2.mut(1, 2) = 0.f;

  MutImageBool equal_mask = isEqualMask(mut_image, mut_image2);
  SOPHUS_ASSERT_EQ(countTrue(equal_mask), 5);

  EXPECT_DEATH(
      SOPHUS_ASSERT_IMAGE_EQ(mut_image, mut_image2), "First failed pixel");
}
