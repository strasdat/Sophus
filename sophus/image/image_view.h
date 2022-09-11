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

#include "sophus/image/image_size.h"

#include <farm_ng/core/logging/logger.h>

namespace sophus {

// Types are largely inspired / derived from Pangolin.

template <class PixelT, template <typename> class AllocatorT>
class MutImage;

template <class PixelT, template <typename> class AllocatorT>
class Image;

/// A view of an (immutable) image, which does not own the data.
///
/// The API of ImageView allows for read-only access. There is an escape hook
/// for write access, see MutImageView::unsafeConstCast below.
///
/// ImageViews are nullable. In that case `this->isEmpty()` is true.
///
///
/// Details on equality comparison, the state of the object, and
/// const-correctness.
///
/// ImageView is a "shallow-compare type" similar to std::span<Pixel const> and
/// std::unique_ptr<Pixel const>. In particular, we define that the state of an
/// ImageView instance consists of the shape of the image ``shape_`` (see
/// ImageShape) and the pointer address to the first pixel ``ptr_``. No
/// public member method can change the pointer nor the shape, hence they are
/// all marked const.
template <class PixelT>
struct ImageView {
  /// Default constructor creates an empty image.
  ImageView() = default;

  /// Creates view from shape and pointer to first pixel.
  ImageView(ImageShape shape, PixelT const* ptr) noexcept
      : shape_(shape), ptr_(ptr) {}

  /// Creates view from image size and pointer to first pixel. The image is
  /// assumed to be contiguous and the pitch is set accordingly.
  ImageView(sophus::ImageSize image_size, PixelT const* ptr) noexcept
      : ImageView(ImageShape::makeFromSize<PixelT>(image_size), ptr) {}

  /// Returns true if view is empty.
  [[nodiscard]] bool isEmpty() const { return this->ptr_ == nullptr; }

  /// Returns true if view is contiguous.
  [[nodiscard]] bool isContiguous() {
    return imageSize().width * sizeof(PixelT) == shape().pitchBytes();
  }

  /// Returns ImageSize.
  /// It is {0,0} if view is empty.
  [[nodiscard]] sophus::ImageSize const& imageSize() const {
    return shape_.imageSize();
  }

  /// Returns ImageShape.
  /// It is {{0,0}, 0} is view is empty.
  [[nodiscard]] ImageShape const& shape() const { return shape_; }

  [[nodiscard]] int width() const { return shape().width(); }
  [[nodiscard]] int height() const { return shape().height(); }
  [[nodiscard]] size_t pitchBytes() const { return shape().pitchBytes(); }

  /// Returns true if u is in [0, width).
  [[nodiscard]] bool colInBounds(int u) const {
    return u >= 0 && u < shape_.width();
  }

  /// Returns true if v is in [0, height).
  [[nodiscard]] bool rowInBounds(int v) const {
    return v >= 0 && v < shape_.height();
  }

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] PixelT const* rowPtr(int v) const {
    return (PixelT*)((uint8_t*)(ptr_) + v * shape_.pitchBytes());
  }

  /// Returns pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  ///
  /// Note:
  ///  * panics if u.v is invalid,
  ///
  /// This is a costly function to call - e.g. when iterating over the whole
  /// image. Use the following instead:
  ///
  /// for (int v=0; v<view.shape().height(); ++v) {
  ///   PixelT const* row = img.rowPtr(v);
  ///   for (int u=0; u<view.shape().width(); ++u) {
  ///     PixetT p = row[u];
  ///   }
  /// }
  [[nodiscard]] PixelT const& checked(int u, int v) const {
    FARM_CHECK(colInBounds(u));
    FARM_CHECK(rowInBounds(v));
    return rowPtr(v)[u];
  }

  [[nodiscard]] PixelT const& checked(Eigen::Vector2i uv) const {
    return checked(uv[0], uv[1]);
  }

  /// Returns pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  /// Silent UB on failure.
  [[nodiscard]] PixelT const& unchecked(int u, int v) const {
    return rowPtr(v)[u];
  }

  [[nodiscard]] PixelT const& unchecked(Eigen::Vector2i uv) const {
    return unchecked(uv[0], uv[1]);
  }

  /// Returns pointer to first pixel.
  [[nodiscard]] PixelT const* ptr() const { return ptr_; }

  /// Returns subview.
  [[nodiscard]] ImageView subview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    FARM_CHECK(colInBounds(uv[0]));
    FARM_CHECK(rowInBounds(uv[1]));
    FARM_CHECK_LE(uv.x() + size.width, shape_.width());
    FARM_CHECK_LE(uv.y() + size.height, shape_.height());
    return ImageView(
        ImageShape::makeFromSizeAndPitch<PixelT>(size, shape_.pitchBytes()),
        rowPtr(uv.y()) + uv.x());
  }

  /// Performs reduction / fold on image view.
  template <class ReduceOpT, class ValT>
  [[nodiscard]] ValT reduce(
      const ReduceOpT& reduce_op, ValT val = ValT{}) const {
    FARM_CHECK(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      const PixelT* p = this->rowPtr(v);
      const PixelT* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p) {
        reduce_op(*p, val);
      }
    }
    return val;
  }

  /// Performs reduction / fold on image view with short circuit condition.
  template <class ShortCircuitReduceOpT, class ValT>
  [[nodiscard]] ValT shortCircuitReduce(
      const ShortCircuitReduceOpT& short_circuit_reduce_op,
      ValT val = ValT{}) const {
    FARM_CHECK(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      const PixelT* p = this->rowPtr(v);
      const PixelT* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p) {
        if (short_circuit_reduce_op(*p, val)) {
          return val;
        }
      }
    }
    return val;
  }

  /// The equality operator is deleted to avoid confusion. Since ImageView is a
  /// "shallow-copy" type, a consistently defined equality would check for
  /// equality of its (shallow) state:
  ///
  ///    ```this->shape_ == rhs.shape() && this->ptr_ == rhs.ptr_````
  ///
  /// However, some users might expect that equality would check for pixel
  /// values equality and return true for identical copies of data blocks.
  ///
  /// Here we follow std::span which also does not offer equality comparions.
  bool operator==(const ImageView& rhs) const = delete;

  /// The in-equality operator is deleted to avoid confusion.
  bool operator!=(const ImageView& rhs) const = delete;

  /// Returns true both views have the same size and contain the same data.
  bool hasSameData(const ImageView& rhs) const {
    if (!(this->imageSize() == rhs.imageSize())) {
      return false;
    }
    for (int v = 0; v < this->shape_.height(); ++v) {
      const PixelT* p = this->rowPtr(v);
      const PixelT* rhs_p = rhs.rowPtr(v);

      const PixelT* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p, ++rhs_p) {
        if (*p != *rhs_p) {
          return false;
        }
      }
    }
    return true;
  }

 protected:
  /// Resets view such that it is empty.
  void setViewToEmpty() { *this = {}; }

  ImageShape shape_ = {};        // NOLINT
  PixelT const* ptr_ = nullptr;  // NOLINT

 private:
  template <class T2T, template <typename> class AllocatorT>
  friend class MutImage;

  template <class T2T, template <typename> class AllocatorT>
  friend class Image;
};

namespace details {
void pitchedCopy(
    uint8_t* dst,
    size_t dst_pitch_bytes,
    const uint8_t* src,
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
/// MutImageView is a "shallow-compare type" similar to std::span<<Pixel> and
/// std::unique_ptr<Pixel>. As ImageView, its state consists of the image shape
/// as well as the pointer address, and comparing those entities establishes
/// equality comparisons. Furthermore, giving mutable access to pixels is
/// considered a const operation, as in
///
///  ```PixelT& checkedMut(int u, int v) const```
///
/// since this merely allows for changing a pixel value, but not its state
/// (data location and layout).
template <class PixelT>
class MutImageView : public ImageView<PixelT> {
 public:
  /// Default constructor creates an empty image.
  MutImageView() = default;

  /// Creates view from shape and pointer to first pixel.
  MutImageView(ImageShape shape, PixelT* ptr) noexcept
      : ImageView<PixelT>(shape, ptr) {}

  /// Creates view from image size and pointer to first pixel. The image is
  /// assumed to be contiguous and the pitch is set accordingly.
  MutImageView(sophus::ImageSize image_size, PixelT* ptr) noexcept
      : ImageView<PixelT>(image_size, ptr) {}

  /// Creates mutable view from view.
  ///
  /// It is the user's responsibility to make sure that the data owned by
  /// the view can be modified safely.
  [[nodiscard]] static MutImageView unsafeConstCast(ImageView<PixelT> view) {
    return MutImageView(view.shape(), const_cast<PixelT*>(view.ptr()));
  }

  /// Returns ImageView(*this).
  ///
  /// Returns non-mut version of view.
  [[nodiscard]] ImageView<PixelT> view() const {
    return ImageView<PixelT>(this->shape(), this->ptr());
  }

  /// Copies data from view into this.
  ///
  /// Preconditions:
  ///  * this->isEmpty() == view.isEmpty()
  ///  * this->size() == view.size()
  ///
  /// No-op if view is empty.
  void copyDataFrom(ImageView<PixelT> view) const {
    FARM_CHECK_EQ(this->isEmpty(), view.isEmpty());

    if (this->isEmpty()) {
      return;
    }
    FARM_CHECK_EQ(this->imageSize(), view.imageSize());
    details::pitchedCopy(
        (uint8_t*)this->ptr(),
        this->shape().pitchBytes(),
        (const uint8_t*)view.ptr(),
        view.shape().pitchBytes(),
        this->imageSize(),
        sizeof(PixelT));
  }

  /// Returns v-th row pointer of mutable pixel.
  [[nodiscard]] PixelT* mutRowPtr(int v) const {
    return (PixelT*)((uint8_t*)(this->ptr()) + v * this->shape_.pitchBytes());
  }

  /// Mutable accessor to pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  ///
  /// Note:
  ///  * panics if u,v is invalid.
  [[nodiscard]] PixelT& checkedMut(int u, int v) const {
    FARM_CHECK(this->colInBounds(u));
    FARM_CHECK(this->rowInBounds(v));
    return mutRowPtr(v)[u];
  }

  [[nodiscard]] PixelT& checkedMut(Eigen::Vector2i uv) const {
    return checkedMut(uv[0], uv[1]);
  }

  /// Mutable accessor to pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  /// Silent UB on failure.
  [[nodiscard]] PixelT& uncheckedMut(int u, int v) const {
    return mutRowPtr(v)[u];
  }

  [[nodiscard]] PixelT& uncheckedMut(Eigen::Vector2i uv) const {
    return uncheckedMut(uv[0], uv[1]);
  }

  /// Mutates each pixel of this with given unary operation
  ///
  /// Preconditions: this must not be empty.
  template <class UnaryOperationT>
  void mutate(const UnaryOperationT& unary_op) const {
    FARM_CHECK(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      PixelT* p = this->mutRowPtr(v);
      const PixelT* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p) {
        *p = unary_op(*p);
      }
    }
  }

  /// Transforms view using unary operation and assigns result to this.
  ///
  /// Preconditions:
  ///   - this must not be empty.
  ///   - this->imageSize() == view.imageSize()
  template <class OtherPixelT, class UnaryOperationT>
  void transformFrom(
      ImageView<OtherPixelT> view, const UnaryOperationT& unary_op) const {
    FARM_CHECK(!this->isEmpty());
    FARM_CHECK_EQ(view.imageSize(), this->imageSize());

    for (int v = 0; v < this->shape_.height(); ++v) {
      PixelT* mut_p = this->mutRowPtr(v);
      OtherPixelT const* p = view.rowPtr(v);
      const OtherPixelT* end_of_row = p + view.shape().width();

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
  template <class LhsPixelT, class RhsPixelT, class BinaryOperationT>
  void transformFrom(
      ImageView<LhsPixelT> lhs,
      ImageView<RhsPixelT> rhs,
      const BinaryOperationT& binary_op) const {
    FARM_CHECK(!this->isEmpty());
    FARM_CHECK_EQ(lhs.imageSize(), this->imageSize());
    FARM_CHECK_EQ(rhs.imageSize(), this->imageSize());

    for (int v = 0; v < this->shape_.height(); ++v) {
      PixelT* mut_p = this->mutRowPtr(v);
      LhsPixelT const* lhs_p = lhs.rowPtr(v);
      RhsPixelT const* rhs_p = rhs.rowPtr(v);

      for (int u = 0; u < this->shape_.width(); ++u) {
        mut_p[u] = binary_op(lhs_p[u], rhs_p[u]);
      }
    }
  }

  /// Populates every pixel of this with val.
  ///
  /// Preconditions: this must not be empty.
  void fill(const PixelT& val) const {
    mutate([&](const PixelT& /*unused*/) { return val; });
  }

  /// Returns pointer of mutable data to first pixel.
  [[nodiscard]] PixelT* mutPtr() const {
    return const_cast<PixelT*>(ImageView<PixelT>::ptr());
  }

  /// Returns mutable subview.
  [[nodiscard]] MutImageView mutSubview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    FARM_CHECK(this->colInBounds(uv[0]));
    FARM_CHECK(this->rowInBounds(uv[1]));
    FARM_CHECK_LE(uv.x() + size.width, this->shape().width());
    FARM_CHECK_LE(uv.y() + size.height, this->shape().height());

    return MutImageView(
        ImageShape::makeFromSizeAndPitch<PixelT>(
            size, this->shape().pitchBytes()),
        this->mutRowPtr(uv.y()) + uv.x());
  }
};

}  // namespace sophus
