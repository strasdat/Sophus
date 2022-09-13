// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image.h"

#include "sophus/image/image_types.h"

#include <farm_ng/core/logging/logger.h>
#include <gtest/gtest.h>

using namespace sophus;

TEST(MutImage, empty) {
  {
    MutImage<float> mut_image;
    FARM_CHECK(mut_image.isEmpty());

    MutImage<float> mut_image_copy = MutImage<float>::makeCopyFrom(mut_image);
    FARM_CHECK(mut_image.isEmpty());
  }
  {
    ImageSize size23(2, 3);
    MutImage<float> mut_image(size23);
    FARM_CHECK(!mut_image.isEmpty());
    FARM_CHECK_EQ(mut_image.imageSize(), size23);
  }
}

TEST(MutImage, create_copy_access) {
  // 1. create new mut image.
  ImageShape shape = ImageShape::makeFromSizeAndPitch<float>(
      {2, 3}, 2 * sizeof(float) + sizeof(float));
  MutImage<float> mut_image(shape);
  mut_image.fill(0.25f);
  FARM_CHECK(!mut_image.isEmpty());
  FARM_CHECK_EQ(mut_image.shape(), shape);

  // 2a. create a copy of it.
  MutImage<float> mut_image_copy = MutImage<float>::makeCopyFrom(mut_image);
  // 2b. test that copy contains the data expected.
  FARM_CHECK(!mut_image.isEmpty());
  FARM_CHECK_IMAGE_EQ(mut_image, mut_image_copy);

  // 3a. create a copy of it.
  MutImage<float> mut_image_copy2 = MutImage<float>::makeCopyFrom(mut_image);
  // 3b. test reset.
  mut_image_copy.reset();
  FARM_CHECK(mut_image_copy.isEmpty());
  // 3c. test swap.
  mut_image_copy2.swap(mut_image_copy);
  FARM_CHECK(!mut_image_copy.isEmpty());
  FARM_CHECK(mut_image_copy2.isEmpty());
  FARM_CHECK_IMAGE_EQ(mut_image, mut_image_copy);

  // 4a. move out of mut_image_copy
  mut_image_copy2 = std::move(mut_image_copy);
  // 4b. test ...
  FARM_CHECK(mut_image_copy.isEmpty());  // NOLINT
  FARM_CHECK_IMAGE_EQ(mut_image, mut_image_copy2);

  // 5a. move out of empty image
  mut_image_copy2 = std::move(mut_image_copy);
  // 5b. test that source and destination are both empty now.
  FARM_CHECK(mut_image_copy.isEmpty());  // NOLINT
  FARM_CHECK(mut_image_copy2.isEmpty());

  // 6a. create a copy of image and move it into mut_image2.
  MutImage<float> mut_image_copy3 = MutImage<float>::makeCopyFrom(mut_image);
  MutImage<float> mut_image2 = std::move(mut_image_copy3);
  // 6b test mut_image == mut_image2
  FARM_CHECK_EQ(mut_image2.imageSize(), mut_image.imageSize());
  for (int v = 0; v < mut_image.imageSize().height; ++v) {
    for (int u = 0; u < mut_image.imageSize().width; ++u) {
      FARM_CHECK_EQ(mut_image2.checked(u, v), mut_image.checked(u, v));
    }
  }

  // 7a. move out of empty mut_image_copy2
  MutImage<float> image3 = std::move(mut_image_copy2);
  // 7b. test that source and destination are both empty now.
  FARM_CHECK(image3.isEmpty());
  FARM_CHECK(mut_image_copy2.isEmpty());  // NOLINT
}

TEST(MutImage, makeFromTransform) {
  ImageShape shape = ImageShape::makeFromSizeAndPitch<float>(
      {2, 3}, 2 * sizeof(float) + sizeof(float));
  MutImage<float> one_image(shape);
  one_image.fill(1.f);

  MutImage3F32 pattern = MutImage3F32::makeFromTransform(
      one_image, [](float v) { return Pixel3F32(v, 0.5f * v, 0.1 * v); });

  for (int v = 0; v < 3; ++v) {
    for (int u = 0; u < 2; ++u) {
      FARM_CHECK_EQ(pattern.checked(u, v), Pixel3F32(1.f, 0.5f, 0.1f));
    }
  }
}

TEST(Image, empty_and_non_empty) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  MutImage<float> copy = MutImage<float>::makeCopyFrom(mut_image);
  FARM_CHECK_EQ(copy.imageSize(), size64);

  Image image(std::move(copy));
  FARM_CHECK(copy.isEmpty());  // NOLINT
  FARM_CHECK_EQ(image.imageSize(), size64);
  FARM_CHECK_EQ(image.useCount(), 1);

  Image empty_image(std::move(copy));
  FARM_CHECK(empty_image.isEmpty());
  FARM_CHECK_EQ(empty_image.useCount(), 0);
}

TEST(Image, shared_ownership) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  Image image(std::move(mut_image));

  Image image2 = image;
  FARM_CHECK_EQ(image.useCount(), 2);
  FARM_CHECK_EQ(image2.useCount(), 2);

  const float* image2_ptr = image2.ptr();
  FARM_CHECK(image2_ptr != nullptr);

  MutImage<float> copy2 = MutImage<float>::makeCopyFrom(std::move(image2));
  FARM_CHECK_EQ(image2.useCount(), 2);  // NOLINT
  FARM_CHECK_NE(size_t(copy2.ptr()), size_t(image2_ptr));

  image2 = image;
  FARM_CHECK_EQ(image.useCount(), 2);
  FARM_CHECK_EQ(image2.useCount(), 2);
}
