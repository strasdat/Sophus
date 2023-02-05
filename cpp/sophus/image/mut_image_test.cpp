// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/mut_image.h"

#include "sophus/image/image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(MutImage, empty) {
  {
    MutImage<float> mut_image;
    SOPHUS_ASSERT(mut_image.isEmpty());

    MutImage<float> mut_image_copy = MutImage<float>::makeCopyFrom(mut_image);
    SOPHUS_ASSERT(mut_image.isEmpty());
  }
  {
    ImageSize size23(2, 3);
    MutImage<float> mut_image(size23);
    SOPHUS_ASSERT(!mut_image.isEmpty());
    SOPHUS_ASSERT_EQ(mut_image.imageSize(), size23);
  }
}

TEST(MutImage, create_copy_access) {
  // 1. create new mut image.
  ImageLayout layout = ImageLayout::makeFromSizeAndPitch<float>(
      {2, 3}, 2 * sizeof(float) + sizeof(float));
  MutImage<float> mut_image(layout);
  mut_image.fill(0.25f);
  SOPHUS_ASSERT(!mut_image.isEmpty());
  SOPHUS_ASSERT_EQ(mut_image.layout(), layout);

  // 2a. create a copy of it.
  MutImage<float> mut_image_copy = MutImage<float>::makeCopyFrom(mut_image);
  // 2b. test that copy contains the data expected.
  SOPHUS_ASSERT(!mut_image.isEmpty());
  SOPHUS_ASSERT_IMAGE_EQ(mut_image, mut_image_copy);

  // 3a. create a copy of it.
  MutImage<float> mut_image_copy2 = MutImage<float>::makeCopyFrom(mut_image);
  // 3b. test reset.
  mut_image_copy.reset();
  SOPHUS_ASSERT(mut_image_copy.isEmpty());
  // 3c. test swap.
  mut_image_copy2.swap(mut_image_copy);
  SOPHUS_ASSERT(!mut_image_copy.isEmpty());
  SOPHUS_ASSERT(mut_image_copy2.isEmpty());
  SOPHUS_ASSERT_IMAGE_EQ(mut_image, mut_image_copy);

  // 4a. move out of mut_image_copy
  mut_image_copy2 = std::move(mut_image_copy);
  // 4b. test ...
  SOPHUS_ASSERT(mut_image_copy.isEmpty());  // NOLINT
  SOPHUS_ASSERT_IMAGE_EQ(mut_image, mut_image_copy2);

  // 5a. move out of empty image
  mut_image_copy2 = std::move(mut_image_copy);
  // 5b. test that source and destination are both empty now.
  SOPHUS_ASSERT(mut_image_copy.isEmpty());  // NOLINT
  SOPHUS_ASSERT(mut_image_copy2.isEmpty());

  // 6a. create a copy of image and move it into mut_image2.
  MutImage<float> mut_image_copy3 = MutImage<float>::makeCopyFrom(mut_image);
  MutImage<float> mut_image2 = std::move(mut_image_copy3);
  // 6b test mut_image == mut_image2
  SOPHUS_ASSERT_EQ(mut_image2.imageSize(), mut_image.imageSize());
  for (int v = 0; v < mut_image.imageSize().height; ++v) {
    for (int u = 0; u < mut_image.imageSize().width; ++u) {
      SOPHUS_ASSERT_EQ(mut_image2(u, v), mut_image(u, v));
    }
  }

  // 7a. move out of empty mut_image_copy2
  MutImage<float> image3 = std::move(mut_image_copy2);
  // 7b. test that source and destination are both empty now.
  SOPHUS_ASSERT(image3.isEmpty());
  SOPHUS_ASSERT(mut_image_copy2.isEmpty());  // NOLINT
}

TEST(MutImage, makeFromTransform) {
  ImageLayout layout = ImageLayout::makeFromSizeAndPitch<float>(
      {2, 3}, 2 * sizeof(float) + sizeof(float));
  MutImage<float> one_image(layout);
  one_image.fill(1.f);

  MutImage3F32 pattern = MutImage3F32::makeFromTransform(
      one_image, [](float v) { return Pixel3F32(v, 0.5f * v, 0.1 * v); });

  for (int v = 0; v < 3; ++v) {
    for (int u = 0; u < 2; ++u) {
      SOPHUS_ASSERT_EQ(pattern(u, v), Pixel3F32(1.f, 0.5f, 0.1f));
    }
  }
}
