// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_view.h"

#include "sophus/image/image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(ImageView, empty) {
  ImageView<float> view;
  SOPHUS_ASSERT(view.isEmpty());
  SOPHUS_ASSERT_EQ(view.imageSize(), ImageSize(0, 0));
  SOPHUS_ASSERT_EQ(view.layout().pitchBytes(), 0u);
  SOPHUS_ASSERT(view.ptr() == nullptr);
}

TEST(ImageView, create_and_access) {
  // clang-format off
  const std::array<uint16_t, 12> data_u16 =
      { 0,  1,  2,
       10, 11, 12,
       20, 21, 22,
       30, 31, 32};
  // clang-format on

  {
    ImageSize image_size(3, 2);
    ImageView<uint16_t> view(image_size, data_u16.data());
    SOPHUS_ASSERT_EQ(image_size, view.imageSize());
    SOPHUS_ASSERT_EQ(
        image_size.width * sizeof(uint16_t), view.layout().pitchBytes());

    SOPHUS_ASSERT(!view.rowInBounds(-1));
    SOPHUS_ASSERT(view.rowInBounds(0));
    SOPHUS_ASSERT(view.rowInBounds(1));
    SOPHUS_ASSERT(!view.rowInBounds(2));

    SOPHUS_ASSERT_EQ(view(0, 0), 0);
    SOPHUS_ASSERT_EQ(view(0, 1), 10);
    SOPHUS_ASSERT_EQ(view(2, 0), 2);

    SOPHUS_ASSERT_EQ(size_t(view.ptr()), size_t(data_u16.data()));

    ImageSize col_view_size(1, 2);
    ImageView<uint16_t> col1 = view.subview({1, 0}, col_view_size);
    SOPHUS_ASSERT_EQ(view.layout().pitchBytes(), col1.layout().pitchBytes());

    SOPHUS_ASSERT_EQ(col1(0, 0), 1);
    SOPHUS_ASSERT_EQ(col1(0, 1), 11);
    SOPHUS_ASSERT_EQ(col_view_size, col1.imageSize());
  }

  ImageSize image_size(1, 2);
  using Pixel2U16 = Eigen::Matrix<uint16_t, 2, 1>;

  ImageView<Pixel2U16> view(
      ImageLayout::makeFromSizeAndPitch<Pixel2U16>(
          image_size, image_size.width * sizeof(Pixel2U16) + sizeof(uint16_t)),
      (Pixel2U16*)data_u16.data());
  SOPHUS_ASSERT_EQ(view.imageSize(), image_size);
  SOPHUS_ASSERT_EQ(view.layout().pitchBytes(), 6);

  SOPHUS_ASSERT_EQ(view(0, 0), Pixel2U16(0, 1));
  SOPHUS_ASSERT_EQ(view(0, 1), Pixel2U16(10, 11));
}
