
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

#include "sophus/calculus/interval.h"
#include "sophus/image/image_size.h"

namespace sophus {

// Types are largely inspired / derived from Pangolin.

template <class TPixel, template <class> class TAllocator>
class MutImage;

template <class TPixel, template <class> class TAllocator>
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
template <class TPixel>
struct ImageView {
  using PixelType = TPixel;

  /// Default constructor creates an empty image.
  ImageView() = default;

  /// Creates view from shape and pointer to first pixel.
  ImageView(ImageShape shape, TPixel const* ptr) noexcept
      : shape_(shape), ptr_(ptr) {}

  /// Creates view from image size and pointer to first pixel. The image is
  /// assumed to be contiguous and the pitch is set accordingly.
  ImageView(sophus::ImageSize image_size, TPixel const* ptr) noexcept
      : ImageView(ImageShape::makeFromSize<TPixel>(image_size), ptr) {}

  /// Returns true if view is empty.
  [[nodiscard]] bool isEmpty() const { return this->ptr_ == nullptr; }

  /// Returns true if view is contiguous.
  [[nodiscard]] bool isContiguous() const {
    return imageSize().width * sizeof(TPixel) == shape().pitchBytes();
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
  [[nodiscard]] TPixel const* rowPtr(int v) const {
    return (TPixel*)((uint8_t*)(ptr_) + v * shape_.pitchBytes());
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
  ///   TPixel const* row = img.rowPtr(v);
  ///   for (int u=0; u<view.shape().width(); ++u) {
  ///     PixetT p = row[u];
  ///   }
  /// }
  [[nodiscard]] TPixel const& checked(int u, int v) const {
    SOPHUS_ASSERT(colInBounds(u), "u,v: {},{}, w x h: {}", u, v, imageSize());
    SOPHUS_ASSERT(rowInBounds(v), "u,v: {},{}, w x h: {}", u, v, imageSize());
    return rowPtr(v)[u];
  }

  [[nodiscard]] TPixel const& checked(Eigen::Vector2i uv) const {
    return checked(uv[0], uv[1]);
  }

  /// Returns pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  /// Silent UB on failure.
  [[nodiscard]] TPixel const& unchecked(int u, int v) const {
    return rowPtr(v)[u];
  }

  [[nodiscard]] TPixel const& unchecked(Eigen::Vector2i uv) const {
    return unchecked(uv[0], uv[1]);
  }

  /// Returns pointer to first pixel.
  [[nodiscard]] TPixel const* ptr() const { return ptr_; }

  /// Returns subview.
  [[nodiscard]] ImageView subview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    SOPHUS_ASSERT(colInBounds(uv[0]));
    SOPHUS_ASSERT(rowInBounds(uv[1]));
    SOPHUS_ASSERT_LE(uv.x() + size.width, shape_.width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, shape_.height());
    return ImageView(
        ImageShape::makeFromSizeAndPitch<TPixel>(size, shape_.pitchBytes()),
        rowPtr(uv.y()) + uv.x());
  }

  /// Performs reduction / fold on image view.
  template <class TFunc>
  void visit(TFunc const& user_function) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p) {
        user_function(*p);
      }
    }
  }

  /// Performs reduction / fold on image view.
  template <class TReduceOp, class TVal>
  [[nodiscard]] TVal reduce(
      TReduceOp const& reduce_op, TVal val = TVal{}) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p) {
        reduce_op(*p, val);
      }
    }
    return val;
  }

  /// Performs reduction / fold on image view with short circuit condition.
  template <class TShortCircuitReduceOp, class TVal>
  [[nodiscard]] TVal shortCircuitReduce(
      TShortCircuitReduceOp const& short_circuit_reduce_op,
      TVal val = TVal{}) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* end_of_row = p + this->shape_.width();
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
  bool operator==(ImageView const& rhs) const = delete;

  /// The in-equality operator is deleted to avoid confusion.
  bool operator!=(ImageView const& rhs) const = delete;

  /// Returns true both views have the same size and contain the same data.
  [[nodiscard]] bool hasSameData(ImageView const& rhs) const {
    if (!(this->imageSize() == rhs.imageSize())) {
      return false;
    }
    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* rhs_p = rhs.rowPtr(v);

      TPixel const* end_of_row = p + this->shape_.width();
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
  TPixel const* ptr_ = nullptr;  // NOLINT

 private:
  template <class TT, template <class> class TAllocator>
  friend class MutImage;

  template <class TT, template <class> class TAllocator>
  friend class Image;
};

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
/// MutImageView is a "shallow-compare type" similar to std::span<<Pixel> and
/// std::unique_ptr<Pixel>. As ImageView, its state consists of the image shape
/// as well as the pointer address, and comparing those entities establishes
/// equality comparisons. Furthermore, giving mutable access to pixels is
/// considered a const operation, as in
///
///  ```TPixel& checkedMut(int u, int v) const```
///
/// since this merely allows for changing a pixel value, but not its state
/// (data location and layout).
template <class TPixel>
class MutImageView : public ImageView<TPixel> {
 public:
  /// Default constructor creates an empty image.
  MutImageView() = default;

  /// Creates view from shape and pointer to first pixel.
  MutImageView(ImageShape shape, TPixel* ptr) noexcept
      : ImageView<TPixel>(shape, ptr) {}

  /// Creates view from image size and pointer to first pixel. The image is
  /// assumed to be contiguous and the pitch is set accordingly.
  MutImageView(sophus::ImageSize image_size, TPixel* ptr) noexcept
      : ImageView<TPixel>(image_size, ptr) {}

  /// Creates mutable view from view.
  ///
  /// It is the user's responsibility to make sure that the data owned by
  /// the view can be modified safely.
  [[nodiscard]] static MutImageView unsafeConstCast(ImageView<TPixel> view) {
    return MutImageView(view.shape(), const_cast<TPixel*>(view.ptr()));
  }

  /// Returns ImageView(*this).
  ///
  /// Returns non-mut version of view.
  [[nodiscard]] ImageView<TPixel> view() const {
    return ImageView<TPixel>(this->shape(), this->ptr());
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
        this->shape().pitchBytes(),
        (uint8_t const*)view.ptr(),
        view.shape().pitchBytes(),
        this->imageSize(),
        sizeof(TPixel));
  }

  /// Returns v-th row pointer of mutable pixel.
  [[nodiscard]] TPixel* rowPtrMut(int v) const {
    return (TPixel*)((uint8_t*)(this->ptr()) + v * this->shape_.pitchBytes());
  }

  /// Mutable accessor to pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  ///
  /// Note:
  ///  * panics if u,v is invalid.
  [[nodiscard]] TPixel& checkedMut(int u, int v) const {
    SOPHUS_ASSERT(this->colInBounds(u));
    SOPHUS_ASSERT(this->rowInBounds(v));
    return rowPtrMut(v)[u];
  }

  [[nodiscard]] TPixel& checkedMut(Eigen::Vector2i uv) const {
    return checkedMut(uv[0], uv[1]);
  }

  /// Mutable accessor to pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  /// Silent UB on failure.
  [[nodiscard]] TPixel& uncheckedMut(int u, int v) const {
    return rowPtrMut(v)[u];
  }

  [[nodiscard]] TPixel& uncheckedMut(Eigen::Vector2i uv) const {
    return uncheckedMut(uv[0], uv[1]);
  }

  /// Mutates each pixel of this with given unary operation
  ///
  /// Preconditions: this must not be empty.
  template <class TUnaryOperation>
  void mutate(TUnaryOperation const& unary_op) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel* p = this->rowPtrMut(v);
      TPixel const* end_of_row = p + this->shape_.width();
      for (; p != end_of_row; ++p) {
        *p = unary_op(*p);
      }
    }
  }

  /// For each pixel in `this` with coordinates (u,v), populates with the user
  /// provided function, evaluated as `uv_op(u,v)`, where u and v are integers
  /// such that u \in [0, width), v \in [0, height)
  ///
  /// Preconditions: this must not be empty.
  template <class TUVOperation>
  void generate(TUVOperation const& uv_op) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel* p = this->rowPtrMut(v);
      TPixel const* end_of_row = p + this->shape_.width();
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

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel* mut_p = this->rowPtrMut(v);
      TOtherPixel const* p = view.rowPtr(v);
      TOtherPixel const* end_of_row = p + view.shape().width();

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

    for (int v = 0; v < this->shape_.height(); ++v) {
      TPixel* mut_p = this->rowPtrMut(v);
      TLhsPixel const* lhs_p = lhs.rowPtr(v);
      TRhsPixel const* rhs_p = rhs.rowPtr(v);

      for (int u = 0; u < this->shape_.width(); ++u) {
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
  [[nodiscard]] TPixel* ptrMut() const {
    return const_cast<TPixel*>(ImageView<TPixel>::ptr());
  }

  /// Returns mutable subview.
  [[nodiscard]] MutImageView mutSubview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    SOPHUS_ASSERT(this->colInBounds(uv[0]));
    SOPHUS_ASSERT(this->rowInBounds(uv[1]));
    SOPHUS_ASSERT_LE(uv.x() + size.width, this->shape().width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, this->shape().height());

    return MutImageView(
        ImageShape::makeFromSizeAndPitch<TPixel>(
            size, this->shape().pitchBytes()),
        this->rowPtrMut(uv.y()) + uv.x());
  }
};

template <class TPixel>
Interval<TPixel> finiteInterval(sophus::ImageView<TPixel> const& image) {
  return image.reduce(
      [](TPixel v, auto& min_max) {
        if (isFinite(v)) {
          min_max.extend(v);
        }
      },
      Interval<TPixel>{});
}

template <class TPixel>
inline Interval<Eigen::Vector2i> imageCoordsInterval(
    sophus::ImageView<TPixel> const& image, int border = 0) {
  // e.g. 10x10 image has valid values [0, ..., 9] in both dimensions
  // a border of 2 would make valid range [2, ..., 7]
  return Interval<Eigen::Vector2i>(Eigen::Vector2i(border, border))
      .extend(Eigen::Vector2i(
          image.width() - border - 1, image.height() - border - 1));
}

}  // namespace sophus
