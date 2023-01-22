// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image.h"

#include "sophus/image/image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(Image, empty_and_non_empty) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  MutImage<float> copy = MutImage<float>::makeCopyFrom(mut_image);
  SOPHUS_ASSERT_EQ(copy.imageSize(), size64);

  Image image(std::move(copy));
  SOPHUS_ASSERT(copy.isEmpty());  // NOLINT
  SOPHUS_ASSERT_EQ(image.imageSize(), size64);
  SOPHUS_ASSERT_EQ(image.useCount(), 1);

  Image empty_image(std::move(copy));
  SOPHUS_ASSERT(empty_image.isEmpty());
  SOPHUS_ASSERT_EQ(empty_image.useCount(), 0);
}

TEST(Image, shared_ownership) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  Image image(std::move(mut_image));

  Image image2 = image;
  SOPHUS_ASSERT_EQ(image.useCount(), 2);
  SOPHUS_ASSERT_EQ(image2.useCount(), 2);

  float const* image2_ptr = image2.ptr();
  SOPHUS_ASSERT(image2_ptr != nullptr);

  MutImage<float> copy2 = MutImage<float>::makeCopyFrom(std::move(image2));
  SOPHUS_ASSERT_EQ(image2.useCount(), 2);  // NOLINT
  SOPHUS_ASSERT_NE(size_t(copy2.ptr()), size_t(image2_ptr));

  image2 = image;
  SOPHUS_ASSERT_EQ(image.useCount(), 2);
  SOPHUS_ASSERT_EQ(image2.useCount(), 2);
}
