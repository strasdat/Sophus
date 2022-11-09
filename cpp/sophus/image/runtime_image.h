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

  inline int bytesPerPixel() const {
    return num_channels * num_bytes_per_pixel_channel;
  }

  template <class TPixel>
  bool is() {
    return fromTemplate<TPixel>() == *this;
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
    class TPredicate = AnyImagePredicate,
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
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-erased image from MutImage.
  /// By design not "explicit".
  template <class TPixel>
  RuntimeImage(MutImage<TPixel>&& image)
      : RuntimeImage(Image<TPixel>(std::move(image))) {
    static_assert(TPredicate::template isTypeValid<TPixel>());
  }

  /// Create type-image image from provided shape and pixel type.
  /// Pixel data is left uninitialized
  RuntimeImage(ImageShape const& shape, RuntimePixelType const& pixel_type)
      : shape_(shape),
        shared_(TAllocator<uint8_t>().allocate(
            shape.height() * shape.pitchBytes())),
        pixel_type_(pixel_type) {
    // TODO: Missing check on ImagePredicate against pixel_type
    //       has to be a runtime check, since we don't know at runtime.

    FARM_CHECK_LE(
        shape.width() * pixel_type.num_channels *
            pixel_type.num_bytes_per_pixel_channel,
        (int)shape.pitchBytes());
  }

  /// Create type-image image from provided size and pixel type.
  /// Pixel data is left uninitialized
  RuntimeImage(ImageSize const& size, RuntimePixelType const& pixel_type)
      : RuntimeImage(
            ImageShape::makeFromSizeAndPitch<uint8_t>(
                size,
                size.width * pixel_type.num_channels *
                    pixel_type.num_bytes_per_pixel_channel),
            pixel_type) {}

  /// Return true is this contains data of type TPixel.
  template <class TPixel>
  [[nodiscard]] bool has() const noexcept {
    RuntimePixelType expected_type = RuntimePixelType::fromTemplate<TPixel>();
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

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] uint8_t const* rawRowPtr(int v) const {
    return ((uint8_t*)(rawPtr()) + v * shape_.pitchBytes());
  }

  /// Returns subview with shared ownership semantics of whole image.
  [[nodiscard]] RuntimeImage subview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    FARM_CHECK(imageSize().contains(uv));
    FARM_CHECK_LE(uv.x() + size.width, shape_.width());
    FARM_CHECK_LE(uv.y() + size.height, shape_.height());

    auto const shape =
        ImageShape::makeFromSizeAndPitchUnchecked(size, pitchBytes());
    const size_t row_offset =
        uv.x() * numBytesPerPixelChannel() * numChannels();
    uint8_t* ptr = shared_.get() + uv.y() * pitchBytes() + row_offset;
    return {shape, std::shared_ptr<uint8_t>(shared_, ptr), pixel_type_};
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
  [[nodiscard]] size_t sizeBytes() const { return height() * pitchBytes(); }

  [[nodiscard]] size_t useCount() const { return shared_.use_count(); }

  [[nodiscard]] uint8_t const* rawPtr() const { return shared_.get(); }

  [[nodiscard]] bool isEmpty() const { return this->rawPtr() == nullptr; }

 private:
  // Private constructor mainly available for constructing sub-views
  RuntimeImage(
      ImageShape shape,
      std::shared_ptr<uint8_t> shared,
      RuntimePixelType pixel_type)
      : shape_(shape), shared_(shared), pixel_type_(pixel_type) {}

  ImageShape shape_ = {};

  std::shared_ptr<uint8_t> shared_;
  RuntimePixelType pixel_type_;
};

/// Image representing any number of channels (>=1) and any floating and
/// unsigned integral channel type.
template <template <typename> class TAllocator = Eigen::aligned_allocator>
using AnyImage = RuntimeImage<AnyImagePredicate, TAllocator>;

template <class TPixelVariant>
struct VariantImagePredicate {
  using PixelVariant = TPixelVariant;

  template <class TPixel>
  static bool constexpr isTypeValid() {
    return farm_ng::has_type_v<TPixel, TPixelVariant>;
  }
};

using IntensityImagePredicate = VariantImagePredicate<std::variant<
    uint8_t,
    uint16_t,
    float,
    Pixel3U8,
    Pixel3U16,
    Pixel3F32,
    Pixel4U8,
    Pixel4U16,
    Pixel4F32>>;

/// Image to represent intensity image / texture as grayscale (=1 channel),
/// RGB (=3 channel ) and RGBA (=4 channel), either uint8_t [0-255],
/// uint16 [0-65535] or float [0.0-1.0] channel type.
template <template <typename> class TAllocator = Eigen::aligned_allocator>
using IntensityImage = RuntimeImage<IntensityImagePredicate, TAllocator>;

namespace detail {
// Call UserFunc with TRuntimeImage cast to the appropriate concrete type
// from the options in PixelTypes...
template <typename TUserFunc, typename TRuntimeImage, typename... TPixelTypes>
struct VisitImpl;

// base case
template <typename TUserFunc, typename TRuntimeImage, typename TPixelType>
struct VisitImpl<TUserFunc, TRuntimeImage, std::variant<TPixelType>> {
  static void visit(TUserFunc&& func, TRuntimeImage const& image) {
    if (image.pixelType().template is<TPixelType>()) {
      func(image.template image<TPixelType>());
    }
  }
};

// inductive case
template <
    typename TUserFunc,
    typename TRuntimeImage,
    typename TPixelType,
    typename... TRest>
struct VisitImpl<TUserFunc, TRuntimeImage, std::variant<TPixelType, TRest...>> {
  static void visit(TUserFunc&& func, TRuntimeImage const& image) {
    if (image.pixelType().template is<TPixelType>()) {
      func(image.template image<TPixelType>());
    } else {
      VisitImpl<TUserFunc, TRuntimeImage, std::variant<TRest...>>::visit(
          std::forward<TUserFunc>(func), image);
    }
  }
};
}  // namespace detail

template <
    typename TUserFunc,
    class TPredicate = IntensityImagePredicate,
    template <typename> class TAllocator = Eigen::aligned_allocator>
void visitImage(
    TUserFunc&& func, RuntimeImage<TPredicate, TAllocator> const& image) {
  using TRuntimeImage = RuntimeImage<TPredicate, TAllocator>;
  detail::
      VisitImpl<TUserFunc, TRuntimeImage, typename TPredicate::PixelVariant>::
          visit(std::forward<TUserFunc>(func), image);
}

}  // namespace sophus
