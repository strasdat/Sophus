// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image.h"
#include "sophus/image/image_types.h"

#include <farm_ng/core/logging/logger.h>
#include <farm_ng/core/misc/variant_utils.h>

#include <variant>

namespace sophus {

template <class TPixel>
struct AnyImagePredicate {
  static bool constexpr kIsTypeValid = true;
};

struct RuntimePixelType {
  NumberType number_type;
  int num_channels;
  int num_bytes_per_pixel_channel;

  template <class TPixel>
  static RuntimePixelType fromTemplate() {
    return RuntimePixelType{
        .number_type =
            std::is_floating_point_v<typename ImageTraits<TPixel>::ChannelT>
                ? NumberType::floating_point
                : NumberType::fixed_point,
        .num_channels = ImageTraits<TPixel>::kNumChannels,
        .num_bytes_per_pixel_channel =
            sizeof(typename ImageTraits<TPixel>::ChannelT)};
  }
};

bool operator==(RuntimePixelType const& lhs, RuntimePixelType const& rhs);

/// Example:
/// RuntimePixelType::fromTemplate<float>() outputs: "1F32";
/// RuntimePixelType::fromTemplate<Eigen::Matrix<uint8_t,4,1>>() outputs:
/// "4U8";
std::ostream& operator<<(std::ostream& os, RuntimePixelType const& type);

/// Type-erased image with shared ownership, and read-only access to pixels.
/// Type is nullable.
template <
    template <typename> class TPredicate = AnyImagePredicate,
    template <typename> class TAllocator = Eigen::aligned_allocator>
class RuntimeImage {
 public:
  /// Empty image.
  RuntimeImage() = default;

  /// Create type-erased image from Image.
  ///
  /// Ownership is shared between RuntimeImage and Image, and hence the
  /// reference count will be increased by one (unless input is empty).
  /// By design not "explicit".
  template <class TPixel>
  RuntimeImage(Image<TPixel, TAllocator> const& image)
      : shape_(image.shape()),
        shared_(image.shared_),
        pixel_type_(RuntimePixelType::fromTemplate<TPixel>()) {
    static_assert(TPredicate<TPixel>::kIsTypeValid);
  }

  /// Create type-erased image from MutImage.
  /// By design not "explicit".
  template <class TPixel>
  RuntimeImage(MutImage<TPixel>&& image)
      : RuntimeImage(Image<TPixel>(std::move(image))) {
    static_assert(TPredicate<TPixel>::kIsTypeValid);
  }

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();
    static_assert(TPredicate<TPixel>::kIsTypeValid);
    return expected_type == pixel_type_;
  }

  /// Returns typed image.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] Image<TPixel, TAllocator> image() const noexcept {
    if (!this->has<TPixel>()) {
      RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();

      FARM_FATAL(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          pixel_type_);
    }

    return Image<TPixel, TAllocator>(
        ImageView<TPixel>(shape_, reinterpret_cast<TPixel*>(shared_.get())),
        shared_);
  }

  template <class TPixel>
  Image<TPixel, TAllocator> reinterpretAs(
      ImageSize reinterpreted_size) const noexcept {
    FARM_CHECK_LE(
        reinterpreted_size.width * sizeof(TPixel), shape().pitch_bytes_);
    FARM_CHECK_LE(reinterpreted_size.height, height());

    FARM_UNIMPLEMENTED();
  }

  [[nodiscard]] RuntimePixelType pixelType() const { return pixel_type_; }

  [[nodiscard]] int numChannels() const { return pixel_type_.num_channels; }

  /// Number of bytes per channel of a single pixel.
  ///
  /// E.g. a pixel of Eigen::Matrix<uint8_t, 3, 1> has 1 byte per channel.
  [[nodiscard]] int numBytesPerPixelChannel() const {
    return pixel_type_.num_bytes_per_pixel_channel;
  }
  [[nodiscard]] NumberType numberType() const {
    return pixel_type_.number_type;
  }

  [[nodiscard]] ImageShape const& shape() const { return shape_; }

  [[nodiscard]] ImageSize const& imageSize() const {
    return shape_.imageSize();
  }

  [[nodiscard]] int width() const { return shape().width(); }
  [[nodiscard]] int height() const { return shape().height(); }
  [[nodiscard]] size_t pitchBytes() const { return shape().pitchBytes(); }

  [[nodiscard]] size_t useCount() const { return shared_.use_count(); }

  [[nodiscard]] uint8_t const* rawPtr() const { return shared_.get(); }

  [[nodiscard]] bool isEmpty() const { return this->rawPtr() == nullptr; }

 private:
  ImageShape shape_ = {};

  std::shared_ptr<uint8_t> shared_;
  RuntimePixelType pixel_type_;
};

/// Image representing any number of channels (>=1) and any floating and
/// unsigned integral channel type.
template <template <typename> class TAllocator = Eigen::aligned_allocator>
using AnyImage = RuntimeImage<AnyImagePredicate, TAllocator>;

template <class TPixel>
struct IntensityImagePredicate {
  static int const kNumChannels = ImageTraits<TPixel>::kNumChannels;
  using ChannelT = typename ImageTraits<TPixel>::ChannelT;
  static bool constexpr kIsTypeValid =
      (kNumChannels == 1 || kNumChannels == 3 || kNumChannels == 4) &&
      (std::is_same_v<ChannelT, uint8_t> ||
       std::is_same_v<ChannelT, uint16_t> || std::is_same_v<ChannelT, float>);
  using Variant = std::variant<
      uint8_t,
      uint16_t,
      float,
      Pixel3U8,
      Pixel3U16,
      Pixel3F32,
      Pixel4U8,
      Pixel4U16,
      Pixel4F32>;
  static_assert(kIsTypeValid == farm_ng::has_type_v<TPixel, Variant>);
};

/// Image to represent intensity image / texture as grayscale (=1 channel),
/// RGB (=3 channel ) and RGBA (=4 channel), either uint8_t [0-255],
/// uint16 [0-65535] or float [0.0-1.0] channel type.
template <template <typename> class TAllocator = Eigen::aligned_allocator>
using IntensityImage = RuntimeImage<IntensityImagePredicate, TAllocator>;

}  // namespace sophus
