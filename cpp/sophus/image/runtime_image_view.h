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

#include <variant>

namespace sophus {

struct AnyImagePredicate {
  template <class TPixel>
  static bool constexpr isTypeValid() {
    return true;
  }
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

  [[nodiscard]] inline int bytesPerPixel() const {
    return num_channels * num_bytes_per_pixel_channel;
  }

  template <class TPixel>
  [[nodiscard]] bool is() {
    return fromTemplate<TPixel>() == *this;
  }
};

bool operator==(RuntimePixelType const& lhs, RuntimePixelType const& rhs);

/// Example:
/// RuntimePixelType::fromTemplate<float>() outputs: "1F32";
/// RuntimePixelType::fromTemplate<Eigen::Matrix<uint8_t,4,1>>() outputs:
/// "4U8";
std::ostream& operator<<(std::ostream& os, RuntimePixelType const& type);

template <class TPredicate = AnyImagePredicate>
class RuntimeImageView {
 public:
  /// Create type-erased image view from ImageView.
  ///
  /// By design not "explicit".
  template <class TPixel>
  RuntimeImageView(ImageView<TPixel> const& image)
      : RuntimeImageView(
            image.shape(),
            RuntimePixelType::fromTemplate<TPixel>(),
            image.ptr()) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  RuntimeImageView(
      ImageShape const& image_shape,
      RuntimePixelType const& pixel_type,
      void const* ptr)
      : shape_(image_shape),
        pixel_type_(pixel_type),
        ptr_(reinterpret_cast<uint8_t const*>(ptr)) {}

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();
    return expected_type == pixel_type_;
  }

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] uint8_t const* rawRowPtr(int v) const {
    return ((uint8_t*)(rawPtr()) + v * shape_.pitchBytes());
  }

  [[nodiscard]] uint8_t const* rawPtr() const { return ptr_; }

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
  [[nodiscard]] size_t sizeBytes() const { return height() * pitchBytes(); }
  [[nodiscard]] bool isEmpty() const { return this->rawPtr() == nullptr; }

  [[nodiscard]] RuntimePixelType pixelType() const { return pixel_type_; }
  [[nodiscard]] int numChannels() const {
    return this->pixel_type_.num_channels;
  }

  /// Returns subview with shared ownership semantics of whole image.
  [[nodiscard]] RuntimeImageView subview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    SOPHUS_ASSERT(imageSize().contains(uv));
    SOPHUS_ASSERT_LE(uv.x() + size.width, this->shape_.width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, this->shape_.height());

    auto const shape =
        ImageShape::makeFromSizeAndPitchUnchecked(size, pitchBytes());
    const size_t row_offset =
        uv.x() * numBytesPerPixelChannel() * numChannels();
    uint8_t const* ptr = this->rawPtr() + uv.y() * pitchBytes() + row_offset;
    return RuntimeImageView{shape, this->pixel_type_, ptr};
  }

  /// Returns typed image.
  ///
  /// Precondition: this->has<TPixel>()
  template <class TPixel>
  [[nodiscard]] ImageView<TPixel> imageView() const noexcept {
    if (!this->has<TPixel>()) {
      RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();

      SOPHUS_PANIC(
          "expected type: {}\n"
          "actual type: {}",
          expected_type,
          this->pixel_type_);
    }

    return ImageView<TPixel>(
        this->shape_, reinterpret_cast<TPixel const*>(ptr_));
  }

 protected:
  RuntimeImageView() = default;

  ImageShape shape_ = {};
  RuntimePixelType pixel_type_;
  uint8_t const* ptr_;
};
}  // namespace sophus
