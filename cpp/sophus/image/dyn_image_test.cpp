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
  Image<float> image(std::move(mut_image));
  AnyImage<> any_image(image);

  SOPHUS_ASSERT_EQ(image.useCount(), 2);
  SOPHUS_ASSERT_EQ(any_image.useCount(), 2);

  SOPHUS_ASSERT_EQ(any_image.numChannels(), 1);
  SOPHUS_ASSERT_EQ(
      any_image.pixelFormat().num_bytes_per_component, sizeof(float));
  SOPHUS_ASSERT_EQ(
      any_image.pixelFormat().number_type, NumberType::floating_point);

  SOPHUS_ASSERT(!any_image.has<uint16_t>());
  SOPHUS_ASSERT(!any_image.has<double>());
  SOPHUS_ASSERT(any_image.has<float>());
  SOPHUS_ASSERT(!(any_image.has<Eigen::Vector3f>()));

  Image<float> image2 = any_image.image<float>();
  SOPHUS_ASSERT_EQ(any_image.useCount(), 3);
  SOPHUS_ASSERT_EQ(image2.useCount(), 3);

  // Getting a mut-view from a shared image will give the power to modify
  // the underlying data. Be careful!
  MutImageView<float> mut_view =
      MutImageView<float>::unsafeConstCast(image2);  // NOLINT
  mut_view.mut(0, 0) = 0.9f;

  SOPHUS_ASSERT_EQ(image(0, 0), 0.9f);
  SOPHUS_ASSERT_EQ(image2(0, 0), 0.9f);
}

TEST(IntensityImage, create_access_and_extract) {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  Image<float> image(std::move(mut_image));
  IntensityImage<> texture(image);

  // Won't compile since IntensityImage can't be uint16_t.
  // SOPHUS_ASSERT(!shared_texture_image.has<uint16_t>());
}

TEST(AnyImage, runtime_type_info) {
  {
    const ImageSize size64{6, 4};
    MutImage<float> mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    SOPHUS_ASSERT_EQ(
        any_image.pixelFormat().number_type, NumberType::floating_point);
    SOPHUS_ASSERT_EQ(any_image.numChannels(), 1);
    SOPHUS_ASSERT_EQ(any_image.pixelFormat().num_bytes_per_component, 4);

    auto maybe_any_image2 = AnyImage<>::tryFromFormat(
        any_image.imageSize(), any_image.pixelFormat());
    SOPHUS_ASSERT(maybe_any_image2);

    auto maybe_any_image3 =
        AnyImage<>::tryFromFormat(any_image.layout(), any_image.pixelFormat());
    SOPHUS_ASSERT(maybe_any_image2);
  }
  {
    const ImageSize size64{6, 4};
    MutImage<uint8_t> mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    SOPHUS_ASSERT_EQ(
        any_image.pixelFormat().number_type, NumberType::fixed_point);
    SOPHUS_ASSERT_EQ(any_image.numChannels(), 1);
    SOPHUS_ASSERT_EQ(any_image.pixelFormat().num_bytes_per_component, 1);
  }
  {
    const ImageSize size64{6, 4};
    MutImage3F32 mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    SOPHUS_ASSERT_EQ(
        any_image.pixelFormat().number_type, NumberType::floating_point);
    SOPHUS_ASSERT_EQ(any_image.numChannels(), 3);
    SOPHUS_ASSERT_EQ(any_image.pixelFormat().num_bytes_per_component, 4);
  }
  {
    const ImageSize size64{6, 4};
    MutImage3U8 mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    SOPHUS_ASSERT_EQ(
        any_image.pixelFormat().number_type, NumberType::fixed_point);
    SOPHUS_ASSERT_EQ(any_image.numChannels(), 3);
    SOPHUS_ASSERT_EQ(any_image.pixelFormat().num_bytes_per_component, 1);
  }
  {
    const ImageSize size64{6, 4};
    MutImage<Eigen::Vector4f> mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    SOPHUS_ASSERT_EQ(
        any_image.pixelFormat().number_type, NumberType::floating_point);
    SOPHUS_ASSERT_EQ(any_image.numChannels(), 4);
    SOPHUS_ASSERT_EQ(any_image.pixelFormat().num_bytes_per_component, 4);
  }
  {
    const ImageSize size64{6, 4};
    MutImage4U8 mut_image(size64);
    AnyImage<> any_image(std::move(mut_image));

    SOPHUS_ASSERT_EQ(
        any_image.pixelFormat().number_type, NumberType::fixed_point);
    SOPHUS_ASSERT_EQ(any_image.numChannels(), 4);
    SOPHUS_ASSERT_EQ(any_image.pixelFormat().num_bytes_per_component, 1);
  }
}

auto sum(ImageView<float> view) -> float {
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

TEST(IntensityImage, visitor) {
  {
    MutImage<float> mut_image(ImageSize(4, 4));
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        mut_image.mut(x, y) = 4 * y + x;
      }
    }

    Image<float> ref_image = std::move(mut_image);
    IntensityImage<> dyn_image = ref_image;

    visitImage(
        [&](auto const& image) {
          using Timg = typename std::remove_reference<decltype(image)>::type;
          using TPixel = typename Timg::Pixel;
          SOPHUS_ASSERT(dyn_image.template has<TPixel>());
          if constexpr (std::is_same_v<TPixel, float>) {
            SOPHUS_ASSERT(image.hasSameData(ref_image));
          }
        },
        dyn_image);
  }

  {
    MutImage<Pixel3U8> mut_image(ImageSize(4, 4));
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        mut_image.mut(x, y) = Pixel3U8(x, y, 1);
      }
    }
    Image<Pixel3U8> ref_image = std::move(mut_image);
    IntensityImage<> dyn_image = ref_image;

    visitImage(
        [&](auto const& image) {
          using Timg = typename std::remove_reference<decltype(image)>::type;
          using TPixel = typename Timg::Pixel;
          SOPHUS_ASSERT(dyn_image.template has<TPixel>());
          if constexpr (std::is_same_v<TPixel, Pixel3U8>) {
            SOPHUS_ASSERT(image.hasSameData(ref_image));
          }
        },
        dyn_image);
  }
  {
    MutImage<Pixel3U8> mut_image(ImageSize(4, 4));
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        mut_image.mut(x, y) = Pixel3U8(x, y, 1);
      }
    }
    Image<Pixel3U8> ref_image = std::move(mut_image);
    IntensityImage<> dyn_image = ref_image;
    visitImage(
        Overload{
            [&](ImageView<float> const& /*unused*/) { SOPHUS_ASSERT(false); },
            [&](ImageView<Pixel3U8> /*unused*/) {
              // Should execute here
            },
            [&](auto const& /*unused*/) { SOPHUS_ASSERT(false); },
        },
        dyn_image);

    visitImage(
        Overload{
            [&](ImageView<float> /*unused*/) { SOPHUS_ASSERT(false); },
            [&](ImageView<uint32_t> /*unused*/) { SOPHUS_ASSERT(false); },
            [&](auto const& /*unused*/) {
              // Should execute here
            },
        },
        dyn_image);
  }

  {
    MutImage<float> mut_image(ImageSize(4, 4));
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        mut_image.mut(x, y) = 4 * y + x;
      }
    }

    Image<float> ref_image = std::move(mut_image);
    IntensityImage<> dyn_image = ref_image;
    IntensityImageView runtime_sub = dyn_image.subview({1, 1}, {2, 2});

    visitImage(
        Overload{
            [&](ImageView<float> /*unused*/) {  // Should execute here
            },
            [&](ImageView<uint32_t> /*unused*/) { SOPHUS_ASSERT(false); },
            [&](auto const& /*unused*/) { SOPHUS_ASSERT(false); },
        },
        runtime_sub);
  }

  {
    MutImage<float> mut_image(ImageSize(4, 4));
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        mut_image.mut(x, y) = 4 * y + x;
      }
    }

    IntensityImageView runtime_sub(mut_image);

    visitImage(
        Overload{
            [&](ImageView<float> /*unused*/) {  // Should execute here
            },
            [&](ImageView<uint32_t> /*unused*/) { SOPHUS_ASSERT(false); },
            [&](auto const& /*unused*/) { SOPHUS_ASSERT(false); },
        },
        runtime_sub);
  }
}
