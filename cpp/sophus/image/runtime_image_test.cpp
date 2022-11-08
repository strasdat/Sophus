// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/runtime_image.h"

#include <farm_ng/core/logging/logger.h>
#include <gtest/gtest.h>

using namespace sophus;

#define FARM_NG_TEST_IMG_EQ(lhs, rhs) /* NOLINT*/            \
  do {                                                       \
    FARM_CHECK_EQ(lhs.imageSize(), rhs.imageSize());         \
    for (int v = 0; v < lhs.imageSize().height; ++v) {       \
      for (int u = 0; u < lhs.imageSize().width; ++u) {      \
        FARM_CHECK_EQ(lhs.checked(u, v), rhs.checked(u, v)); \
      }                                                      \
    }                                                        \
  } while (false)

TEST(AnyImage, create_access_and_extract) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  Image<float> image(std::move(mut_image));
  AnyImage<> any_image(image);

  FARM_CHECK_EQ(image.useCount(), 2);
  FARM_CHECK_EQ(any_image.useCount(), 2);

  FARM_CHECK_EQ(any_image.numChannels(), 1);
  FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), sizeof(float));
  FARM_CHECK_EQ(any_image.numberType(), NumberType::floating_point);

  FARM_CHECK(!any_image.has<uint16_t>());
  FARM_CHECK(!any_image.has<double>());
  FARM_CHECK(any_image.has<float>());
  FARM_CHECK(!(any_image.has<Eigen::Vector3f>()));

  Image<float> image2 = any_image.image<float>();
  FARM_CHECK_EQ(any_image.useCount(), 3);
  FARM_CHECK_EQ(image2.useCount(), 3);

  // Getting a mut-view from a shared image will give the power to modify
  // the underlying data. Be careful!
  MutImageView<float> mut_view =
      MutImageView<float>::unsafeConstCast(image2);  // NOLINT
  mut_view.checkedMut(0, 0) = 0.9f;

  FARM_CHECK_EQ(image.checked(0, 0), 0.9f);
  FARM_CHECK_EQ(image2.checked(0, 0), 0.9f);
}

TEST(IntensityImage, create_access_and_extract) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  Image<float> image(std::move(mut_image));
  IntensityImage<> texture(image);

  // Won't compile since IntensityImage can't be uint16_t.
  // FARM_CHECK(!shared_texture_image.has<uint16_t>());
}

TEST(AnyImage, runtime_type_info) {
  {
    const ImageSize size64{6, 4};
    MutImage<float> mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    FARM_CHECK_EQ(any_image.numberType(), NumberType::floating_point);
    FARM_CHECK_EQ(any_image.numChannels(), 1);
    FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), 4);
  }
  {
    const ImageSize size64{6, 4};
    MutImage<uint8_t> mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    FARM_CHECK_EQ(any_image.numberType(), NumberType::fixed_point);
    FARM_CHECK_EQ(any_image.numChannels(), 1);
    FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), 1);
  }
  {
    const ImageSize size64{6, 4};
    MutImage3F32 mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    FARM_CHECK_EQ(any_image.numberType(), NumberType::floating_point);
    FARM_CHECK_EQ(any_image.numChannels(), 3);
    FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), 4);
  }
  {
    const ImageSize size64{6, 4};
    MutImage3U8 mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    FARM_CHECK_EQ(any_image.numberType(), NumberType::fixed_point);
    FARM_CHECK_EQ(any_image.numChannels(), 3);
    FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), 1);
  }
  {
    const ImageSize size64{6, 4};
    MutImage<Eigen::Vector4f> mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    FARM_CHECK_EQ(any_image.numberType(), NumberType::floating_point);
    FARM_CHECK_EQ(any_image.numChannels(), 4);
    FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), 4);
  }
  {
    const ImageSize size64{6, 4};
    MutImage4U8 mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    FARM_CHECK_EQ(any_image.numberType(), NumberType::fixed_point);
    FARM_CHECK_EQ(any_image.numChannels(), 4);
    FARM_CHECK_EQ(any_image.numBytesPerPixelChannel(), 1);
  }
}

float sum(ImageView<float> view) {
  float s = 0.0;
  for (int v = 0; v < view.height(); ++v) {
    float const* row = view.rowPtr(v);
    for (int u = 0; u < view.width(); ++u) {
      float p = row[u];
      s += p;
    }
  }
  return s;
}

void plusOne(MutImageView<float> mut_view) {
  for (int v = 0; v < mut_view.height(); ++v) {
    float* row = mut_view.rowPtrMut(v);
    for (int u = 0; u < mut_view.width(); ++u) {
      float& p = row[u];
      p += 1.f;
    }
  }
}

TEST(IntensityImage, subview) {
  MutImage<float> mut_image(ImageSize(4, 4));
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      mut_image.checkedMut(x, y) = 4 * y + x;
    }
  }

  Image<float> ref_image = std::move(mut_image);
  IntensityImage<> runtime_image = ref_image;
  IntensityImage<> runtime_sub = runtime_image.subview({1, 1}, {2, 2});
  FARM_CHECK_EQ(runtime_sub.width(), 2);
  FARM_CHECK_EQ(runtime_sub.height(), 2);

  {
    Image<float> sub = runtime_sub.image<float>();

    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        FARM_CHECK_EQ(sub.checked(x, y), ref_image.checked(x + 1, y + 1));
      }
    }
  }
}

TEST(ClassHierarchy, call_function) {
  // The sum function takes a ImageView<float> as input. Hence we can pass in:
  //  - ImageView<float>
  //  - MutImageView<float>
  //  - MutImage<float>
  //  - Image<float>

  // clang-format off
  std::array<float, 6> data =
      { 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f };
  // clang-format on

  ImageSize image_size(3, 2);
  ImageView<float> view(image_size, data.data());
  float s = sum(view);
  FARM_CHECK_EQ(s, 3.f);

  MutImageView<float> mut_view(image_size, data.data());
  s = sum(view);
  FARM_CHECK_EQ(s, 3.f);

  MutImage<float> mut_image(image_size);
  mut_image.fill(0.5f);
  s = sum(mut_image);
  FARM_CHECK_EQ(s, 3.f);

  Image<float> image = Image<float>::makeCopyFrom(mut_image);
  s = sum(image);
  FARM_CHECK_EQ(s, 3.f);

  // The plusOne function takes a MutImageView<float> as input. Hence we can
  // pass in:
  //  - MutImageView<float>
  //  - MutImage<float>
  plusOne(mut_view);
  s = sum(mut_view);
  FARM_CHECK_EQ(s, 9.f);

  plusOne(mut_image);
  s = sum(mut_image);
  FARM_CHECK_EQ(s, 9.f);

  // won't compile, since ImageView is not a MutImageView:
  // plusOne(view);

  // won't compile, since Image is a ImageView but not a MutImageView:
  // plusOne(image);
}
