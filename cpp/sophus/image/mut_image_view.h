
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// ImageView and MutImageView, non owning images types.
///
/// See image.h for Image and MutImage, owning images types.
///
/// Note that it is a conscious API decision to follow "shallow-compare" type
/// semantic for ImageView, MutImageView, Image and MutImage. Similar
/// "shallow-compare" types are: std::span (shallow-compare reference type), and
/// std::unique_ptr (shallow compare unique ownership type).
/// This is in contrast to regular types such as std::vector, std::string and
/// reference types which mimic regular type semantic such as std::string_view.
/// Also see https://abseil.io/blog/20180531-regular-types.

#pragma once

#include "sophus/calculus/region.h"
#include "sophus/image/image_view.h"

namespace sophus {

namespace details {

void pitchedCopy(
    uint8_t* dst,
    size_t dst_pitch_bytes,
    uint8_t const* src,
    size_t src_pitch_bytes,
    sophus::ImageSize size,
    uint8_t size_of_pixel);
}

/// View of a mutable image, which does not own the data.
///
/// The API of MutImageView allows for read and write access.
///
/// MutImageView is nullable. In that case `this->isEmpty()` is true.
///
///
/// Details on equality comparison, the state of the object, and
/// const-correctness.
///
/// MutImageView is a "shallow-compare type" similar to std::span<Pixel> and
/// std::unique_ptr<Pixel>. As ImageView, its state consists of the image layout
/// as well as the pointer address, and comparing those entities establishes
/// equality comparisons. Furthermore, giving mutable access to pixels is
/// considered a const operation, as in
///
///  ```TPixel& mut(int u, int v) const```
///
/// since this merely allows for changing a pixel value, but not its state
/// (data location and layout).
template <class TPixel>
class MutImageView : public ImageView<TPixel> {
 public:
  /// Default constructor creates an empty image.
  MutImageView() = default;

  /// Creates view from layout and pointer to first pixel.
  MutImageView(ImageLayout layout, TPixel* ptr) noexcept
      : ImageView<TPixel>(layout, ptr) {}

  /// Creates view from image size and pointer to first pixel. The image is
  /// assumed to be contiguous and the pitch is set accordingly.
  MutImageView(sophus::ImageSize image_size, TPixel* ptr) noexcept
      : ImageView<TPixel>(image_size, ptr) {}

  /// Creates mutable view from view.
  ///
  /// It is the user's responsibility to make sure that the data owned by
  /// the view can be modified safely.
  [[nodiscard]] static auto unsafeConstCast(ImageView<TPixel> view)
      -> MutImageView {
    return MutImageView(view.layout(), const_cast<TPixel*>(view.ptr()));
  }

  /// Returns ImageView(*this).
  ///
  /// Returns non-mut version of view.
  [[nodiscard]] auto view() const -> ImageView<TPixel> {
    return ImageView<TPixel>(this->layout(), this->ptr());
  }

  /// Copies data from view into this.
  ///
  /// Preconditions:
  ///  * this->isEmpty() == view.isEmpty()
  ///  * this->size() == view.size()
  ///
  /// No-op if view is empty.
  void copyDataFrom(ImageView<TPixel> view) const {
    SOPHUS_ASSERT_EQ(this->isEmpty(), view.isEmpty());

    if (this->isEmpty()) {
      return;
    }
    SOPHUS_ASSERT_EQ(this->imageSize(), view.imageSize());
    details::pitchedCopy(
        (uint8_t*)this->ptr(),
        this->layout().pitchBytes(),
        (uint8_t const*)view.ptr(),
        view.layout().pitchBytes(),
        this->imageSize(),
        sizeof(TPixel));
  }

  /// Returns v-th row pointer of mutable pixel.
  [[nodiscard]] auto rowPtrMut(int v) const -> TPixel* {
    return (TPixel*)((uint8_t*)(this->ptr()) + v * this->layout_.pitchBytes());
  }

  /// Mutable accessor to pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  /// Silent UB on failure.
  [[nodiscard]] auto mut(int u, int v) const -> TPixel& {
    return rowPtrMut(v)[u];
  }

  [[nodiscard]] auto mut(Eigen::Vector2i uv) const -> TPixel& {
    return mut(uv[0], uv[1]);
  }

  /// Mutates each pixel of this with given unary operation
  ///
  /// Preconditions: this must not be empty.
  template <class TUnaryOperation>
  void mutate(TUnaryOperation const& unary_op) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel* p = this->rowPtrMut(v);
      TPixel const* end_of_row = p + this->layout_.width();
      for (; p != end_of_row; ++p) {
        *p = unary_op(*p);
      }
    }
  }

  /// For each pixel in `this` with coordinates (u,v), populates with the user
  /// provided function, evaluated as `uv_op(u,v)`, where u and v are integers
  /// such that u in [0, width), v in [0, height)
  ///
  /// Preconditions: this must not be empty.
  template <class TUVOperation>
  void generate(TUVOperation const& uv_op) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel* p = this->rowPtrMut(v);
      TPixel const* end_of_row = p + this->layout_.width();
      for (int u = 0; p != end_of_row; ++p, ++u) {
        *p = uv_op(u, v);
      }
    }
  }

  /// Transforms view using unary operation and assigns result to this.
  ///
  /// Preconditions:
  ///   - this must not be empty.
  ///   - this->imageSize() == view.imageSize()
  template <class TOtherPixel, class TUnaryOperation>
  void transformFrom(
      ImageView<TOtherPixel> view, TUnaryOperation const& unary_op) const {
    SOPHUS_ASSERT(!this->isEmpty());
    SOPHUS_ASSERT_EQ(view.imageSize(), this->imageSize());

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel* mut_p = this->rowPtrMut(v);
      TOtherPixel const* p = view.rowPtr(v);
      TOtherPixel const* end_of_row = p + view.layout().width();

      for (; p != end_of_row; ++p, ++mut_p) {
        *mut_p = unary_op(*p);
      }
    }
  }

  /// Transforms two views using binary operation and assigns result to this.
  ///
  /// Preconditions:
  ///   - this must not be empty.
  ///   - this->imageSize() == lhs.imageSize() == rhs.imageSize()
  template <class TLhsPixel, class TRhsPixel, class TBinaryOperation>
  void transformFrom(
      ImageView<TLhsPixel> lhs,
      ImageView<TRhsPixel> rhs,
      TBinaryOperation const& binary_op) const {
    SOPHUS_ASSERT(!this->isEmpty());
    SOPHUS_ASSERT_EQ(lhs.imageSize(), this->imageSize());
    SOPHUS_ASSERT_EQ(rhs.imageSize(), this->imageSize());

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel* mut_p = this->rowPtrMut(v);
      TLhsPixel const* lhs_p = lhs.rowPtr(v);
      TRhsPixel const* rhs_p = rhs.rowPtr(v);

      for (int u = 0; u < this->layout_.width(); ++u) {
        mut_p[u] = binary_op(lhs_p[u], rhs_p[u]);
      }
    }
  }

  /// Populates every pixel of this with val.
  ///
  /// Preconditions: this must not be empty.
  void fill(TPixel const& val) const {
    mutate([&](TPixel const& /*unused*/) { return val; });
  }

  /// Returns pointer of mutable data to first pixel.
  [[nodiscard]] auto ptrMut() const -> TPixel* {
    return const_cast<TPixel*>(ImageView<TPixel>::ptr());
  }

  /// Returns mutable subview.
  [[nodiscard]] auto mutSubview(
      Eigen::Vector2i uv, sophus::ImageSize size) const -> MutImageView {
    SOPHUS_ASSERT(this->colInBounds(uv[0]));
    SOPHUS_ASSERT(this->rowInBounds(uv[1]));
    SOPHUS_ASSERT_LE(uv.x() + size.width, this->layout().width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, this->layout().height());

    return MutImageView(
        ImageLayout::makeFromSizeAndPitch<TPixel>(
            size, this->layout().pitchBytes()),
        this->rowPtrMut(uv.y()) + uv.x());
  }
};

namespace details {

template <class TPixel>
auto checkedPixelAccessMut(
    MutImageView<TPixel> const& view,
    int u,
    int v,
    std::string const& file,
    int line,
    std::string const& str) -> TPixel& {
  if (!view.colInBounds(u) || !view.rowInBounds(v)) {
    FARM_IMPL_LOG_PRINTLN("[SOPHUS_PIXEL_MUT in {}:{}]", file, line);
    FARM_IMPL_LOG_PRINTLN(
        "pixel `{},{}` not in image with size {} x {}",
        u,
        v,
        view.imageSize().width,
        view.imageSize().height);
    if (!str.empty()) {
      ::fmt::print(stderr, "{}", str);
    }
    FARM_IMPL_ABORT();
  }
  return view.mut(u, v);
}
}  // namespace details

}  // namespace sophus

#define SOPHUS_PIXEL_MUT(img, u, v, ...)    \
  ::sophus::details::checkedPixelAccessMut( \
      img, u, v, __FILE__, __LINE__, SOPHUS_FORMAT(__VA_ARGS__))
