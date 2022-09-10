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

template <class PixelT>
struct AnyImagePredicate {
  static constexpr bool kIsTypeValid = true;
};

struct RuntimePixelType {
  NumberType number_type;
  int num_channels;
  int num_bytes_per_pixel_channel;

  template <class PixelT>
  static RuntimePixelType fromTemplate() {
    return RuntimePixelType{
        .number_type =
            std::is_floating_point_v<typename ImageTraits<PixelT>::ChannelT>
                ? NumberType::floating_point
                : NumberType::fixed_point,
        .num_channels = ImageTraits<PixelT>::kNumChannels,
        .num_bytes_per_pixel_channel =
            sizeof(typename ImageTraits<PixelT>::ChannelT)};
  }
};

bool operator==(const RuntimePixelType& lhs, const RuntimePixelType& rhs);

/// Example:
/// RuntimePixelType::fromTemplate<float>() outputs: "1F32";
/// RuntimePixelType::fromTemplate<Eigen::Matrix<uint8_t,4,1>>() outputs:
/// "4U8";
std::ostream& operator<<(std::ostream& os, const RuntimePixelType& type);

/// Type-erased image with shared ownership, and read-only access to pixels.
/// Type is nullable.
template <
    template <typename> class PredicateT = AnyImagePredicate,
    template <typename> class AllocatorT = Eigen::aligned_allocator>
class RuntimeImage {
 public:
  /// Empty image.
  RuntimeImage() = default;

  /// Create type-erased image from Image.
  ///
  /// Ownership is shared between RuntimeImage and Image, and hence the
  /// reference count will be increased by one (unless input is empty).
  /// By design not "explicit".
  template <class PixelT>
  RuntimeImage(Image<PixelT, AllocatorT> const& image)
      : shape_(image.shape()),
        shared_(image.shared_),
        pixel_type_(RuntimePixelType::fromTemplate<PixelT>()) {
    static_assert(PredicateT<PixelT>::kIsTypeValid);
  }

  /// Create type-erased image from MutImage.
  /// By design not "explicit".
  template <class PixelT>
  RuntimeImage(MutImage<PixelT>&& image)
      : RuntimeImage(Image<PixelT>(std::move(image))) {
    static_assert(PredicateT<PixelT>::kIsTypeValid);
  }

  /// Return true is this contains data of type PixelT.
  template <class PixelT>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<PixelT>();
    static_assert(PredicateT<PixelT>::kIsTypeValid);
    return expected_type == pixel_type_;
  }

  /// Returns typed image.
  ///
  /// Precondition: this->has<PixelT>()
  template <class PixelT>
  [[nodiscard]] Image<PixelT, AllocatorT> image() const noexcept {
    if (!this->has<PixelT>()) {
      RuntimePixelType expected_type = RuntimePixelType::fromTemplate<PixelT>();

      FARM_FATAL(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          pixel_type_);
    }

    return Image<PixelT, AllocatorT>(
        ImageView<PixelT>(shape_, reinterpret_cast<PixelT*>(shared_.get())),
        shared_);
  }

  template <class PixelT>
  Image<PixelT, AllocatorT> reinterpretAs(
      ImageSize reinterpreted_size) const noexcept {
    FARM_CHECK_LE(
        reinterpreted_size.width * sizeof(PixelT), shape().pitch_bytes_);
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

  [[nodiscard]] const uint8_t* rawPtr() const { return shared_.get(); }

  [[nodiscard]] bool isEmpty() const { return this->rawPtr() == nullptr; }

 private:
  ImageShape shape_ = {};

  std::shared_ptr<uint8_t> shared_;
  RuntimePixelType pixel_type_;
};

/// Image representing any number of channels (>=1) and any floating and
/// unsigned integral channel type.
template <template <typename> class AllocatorT = Eigen::aligned_allocator>
using AnyImage = RuntimeImage<AnyImagePredicate, AllocatorT>;

template <class PixelT>
struct IntensityImagePredicate {
  static const int kNumChannels = ImageTraits<PixelT>::kNumChannels;
  using ChannelT = typename ImageTraits<PixelT>::ChannelT;
  static constexpr bool kIsTypeValid =
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
  static_assert(kIsTypeValid == farm_ng::has_type_v<PixelT, Variant>);
};

/// Image to represent intensity image / texture as grayscale (=1 channel),
/// RGB (=3 channel ) and RGBA (=4 channel), either uint8_t [0-255],
/// uint16 [0-65535] or float [0.0-1.0] channel type.
template <template <typename> class AllocatorT = Eigen::aligned_allocator>
using IntensityImage = RuntimeImage<IntensityImagePredicate, AllocatorT>;

}  // namespace sophus
