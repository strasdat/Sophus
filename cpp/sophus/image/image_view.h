
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
#include "sophus/image/layout.h"

namespace sophus {

// Types are largely inspired / derived from Pangolin.

template <class TPixel, class TAllocator>
class MutImage;

template <class TPixel, class TAllocator>
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
/// ImageView instance consists of the layout of the image ``layout_`` (see
/// ImageLayout) and the pointer address to the first pixel ``ptr_``. No
/// public member method can change the pointer nor the layout, hence they are
/// all marked const.
template <class TPixel>
struct ImageView {
  using pixelFormat = TPixel;

  /// Default constructor creates an empty image.
  ImageView() = default;

  /// Creates view from layout and pointer to first pixel.
  ImageView(ImageLayout layout, TPixel const* ptr) noexcept
      : layout_(layout), ptr_(ptr) {}

  /// Creates view from image size and pointer to first pixel. The image is
  /// assumed to be contiguous and the pitch is set accordingly.
  ImageView(sophus::ImageSize image_size, TPixel const* ptr) noexcept
      : ImageView(ImageLayout::makeFromSize<TPixel>(image_size), ptr) {}

  /// Returns true if view is empty.
  [[nodiscard]] bool isEmpty() const { return this->ptr_ == nullptr; }

  /// Returns true if view is contiguous.
  [[nodiscard]] bool isContiguous() const {
    return imageSize().width * sizeof(TPixel) == layout().pitchBytes();
  }

  /// Returns ImageSize.
  /// It is {0,0} if view is empty.
  [[nodiscard]] sophus::ImageSize const& imageSize() const {
    return layout_.imageSize();
  }

  /// Returns ImageLayout.
  /// It is {{0,0}, 0} is view is empty.
  [[nodiscard]] ImageLayout const& layout() const { return layout_; }

  [[nodiscard]] int width() const { return layout().width(); }
  [[nodiscard]] int height() const { return layout().height(); }
  [[nodiscard]] size_t pitchBytes() const { return layout().pitchBytes(); }

  /// Returns true if u is in [0, width).
  [[nodiscard]] bool colInBounds(int u) const {
    return u >= 0 && u < layout_.width();
  }

  /// Returns true if v is in [0, height).
  [[nodiscard]] bool rowInBounds(int v) const {
    return v >= 0 && v < layout_.height();
  }

  /// Returns v-th row pointer.
  ///
  /// Precondition: v must be in [0, height).
  [[nodiscard]] TPixel const* rowPtr(int v) const {
    return (TPixel*)((uint8_t*)(ptr_) + v * layout_.pitchBytes());
  }

  /// Returns pixel u, v.
  ///
  /// Precondition: u must be in [0, width) and v must be in [0, height).
  ///
  /// Note:
  ///  * No panic if u.v is invalid,
  ///
  /// This is not the most necessarily the efficient function to call - e.g.
  /// when iterating over the whole image. Use the following instead:
  ///
  /// for (int v=0; v<view.layout().height(); ++v) {
  ///   TPixel const* row = img.rowPtr(v);
  ///   for (int u=0; u<view.layout().width(); ++u) {
  ///     PixetT p = row[u];
  ///   }
  /// }
  [[nodiscard]] TPixel const& operator()(int u, int v) const {
    return rowPtr(v)[u];
  }

  [[nodiscard]] TPixel const& operator()(Eigen::Vector2i uv) const {
    return this->operator()(uv[0], uv[1]);
  }

  /// Returns pointer to first pixel.
  [[nodiscard]] TPixel const* ptr() const { return ptr_; }

  /// Returns subview.
  [[nodiscard]] ImageView subview(
      Eigen::Vector2i uv, sophus::ImageSize size) const {
    SOPHUS_ASSERT(colInBounds(uv[0]));
    SOPHUS_ASSERT(rowInBounds(uv[1]));
    SOPHUS_ASSERT_LE(uv.x() + size.width, layout_.width());
    SOPHUS_ASSERT_LE(uv.y() + size.height, layout_.height());
    return ImageView(
        ImageLayout::makeFromSizeAndPitch<TPixel>(size, layout_.pitchBytes()),
        rowPtr(uv.y()) + uv.x());
  }

  /// Performs reduction / fold on image view.
  template <class TFunc>
  void visit(TFunc const& user_function) const {
    SOPHUS_ASSERT(!this->isEmpty());

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* end_of_row = p + this->layout_.width();
      for (; p != end_of_row; ++p) {
        user_function(*p);
      }
    }
  }

  /// Performs reduction / fold on image view.
  template <class TReduceOp, class TVal>
  [[nodiscard]] TVal reduce(
      TReduceOp const& reduce_op, TVal val = TVal{}) const {
    SOPHUS_ASSERT(!this->isEmpty());  // NOLINT

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* end_of_row = p + this->layout_.width();
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

    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* end_of_row = p + this->layout_.width();
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
  ///    ```this->layout_ == rhs.layout() && this->ptr_ == rhs.ptr_````
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
    for (int v = 0; v < this->layout_.height(); ++v) {
      TPixel const* p = this->rowPtr(v);
      TPixel const* rhs_p = rhs.rowPtr(v);

      TPixel const* end_of_row = p + this->layout_.width();
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

  ImageLayout layout_ = {};      // NOLINT
  TPixel const* ptr_ = nullptr;  // NOLINT

 private:
  template <class TT, class TAllocator>
  friend class MutImage;

  template <class TT, class TAllocator>
  friend class Image;
};

namespace details {

template <class TPixel>
Region<TPixel> finiteInterval(sophus::ImageView<TPixel> const& image) {
  return image.reduce(
      [](TPixel v, auto& min_max) {
        if (isFinite(v)) {
          min_max.extend(v);
        }
      },
      Region<TPixel>{});
}

// TODO: make member function?
template <class TPixel>
inline Region2I imageCoordsInterval(
    sophus::ImageView<TPixel> const& image, int border = 0) {
  return imageCoordsInterval(image.imageSize(), border);
}

template <class TPixel>
[[nodiscard]] auto checkedPixelAccess(
    ImageView<TPixel> const& view,
    int u,
    int v,
    std::string const& file,
    int line,
    std::string const& str) -> TPixel const& {
  if (!view.colInBounds(u) || !view.rowInBounds(v)) {
    FARM_IMPL_LOG_PRINTLN("[SOPHUS_PIXEL in {}:{}]", file, line);
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
  return view(u, v);
}
}  // namespace details

}  // namespace sophus

#define SOPHUS_PIXEL_MUT(img, u, v, ...)    \
  ::sophus::details::checkedPixelAccessMut( \
      img, u, v, __FILE__, __LINE__, SOPHUS_FORMAT(__VA_ARGS__))
