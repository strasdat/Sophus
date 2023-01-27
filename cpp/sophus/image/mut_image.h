// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// Image MutImage, owning images types.
///
/// Note that it is a conscious API decision to follow "shallow-compare" type
/// semantic for ImageView, MutImageView, Image and MutImage. See image_view.h
/// for details.
#pragma once

#include "sophus/common/enum.h"
#include "sophus/image/mut_image_view.h"

#include <optional>

namespace sophus {

// Types are largely inspired / derived from Pangolin.

template <class TPixel, template <class> class TAllocator>
class Image;

struct UniqueDataAreaDeleter {
  virtual ~UniqueDataAreaDeleter() {}

  /// Leaking behaviour in base implementation
  virtual void operator()(uint8_t* p) const {}
};

using UniqueDataArea = std::unique_ptr<uint8_t, UniqueDataAreaDeleter>;

/// A image with write access to pixels and exclusive ownership. There is no
/// copy constr / copy assignment, but move constr / assignment.
///
/// Content from a MutImage can be moved into an Image.
///
/// Type is nullable. In that case `this->isEmpty()` is true.
///
/// Similar to Pangolin::ManagedImage.
template <
    class TPixel,
    template <class> class TAllocator = Eigen::aligned_allocator>
class MutImage : public MutImageView<TPixel> {
 public:
  struct TypedDeleterImpl {
    TypedDeleterImpl(size_t num_bytes) : num_bytes(num_bytes) {}

    void operator()(TPixel* p) const {
      if (p != nullptr) {
        TAllocator<TPixel>().deallocate(p, num_bytes / sizeof(TPixel));
      }
    }

    size_t num_bytes;
  };

  /// Deleter for MutImage.
  struct Deleter : UniqueDataAreaDeleter {
    Deleter() = default;
    Deleter(TypedDeleterImpl image_deleter) : image_deleter(image_deleter) {}

    void operator()(uint8_t* p) const final {
      if (image_deleter) {
        (*image_deleter)(reinterpret_cast<TPixel*>(p));
      }
    }

    std::optional<TypedDeleterImpl> image_deleter;
  };

  template <class TT, template <class> class TAllocator2T>
  friend class Image;

  /// Constructs empty image.
  MutImage() : unique_(nullptr) {}

  /// Creates new image with given shape.
  ///
  /// If shape is not empty, memory allocation will happen.
  explicit MutImage(ImageShape shape)
      : MutImageView<TPixel>(shape, nullptr), unique_(nullptr) {
    if (shape.sizeBytes() != 0u) {
      SOPHUS_ASSERT_EQ(shape.sizeBytes() % sizeof(TPixel), 0);
      this->unique_ = UniqueDataArea(
          (uint8_t*)(TAllocator<TPixel>().allocate(
              shape.sizeBytes() / sizeof(TPixel))),
          Deleter(TypedDeleterImpl(shape.sizeBytes())));
    }
    this->ptr_ = reinterpret_cast<TPixel*>(this->unique_.get());
  }

  /// Creates new contiguous image with given size.
  ///
  /// If shape is not empty, memory allocation will happen.
  explicit MutImage(sophus::ImageSize size)
      : MutImage(ImageShape::makeFromSize<TPixel>(size)) {}

  /// Creates contiguous copy from view.
  ///
  /// If view is not empty, memory allocation will happen.
  [[nodiscard]] static MutImage makeCopyFrom(ImageView<TPixel> const& view) {
    MutImage image(view.imageSize());
    image.copyDataFrom(view);
    return image;
  }

  /// Creates new MutImage given view and unary transform function.
  ///
  /// mut_image(u, v) = unary_op(view(u, v));
  template <class TOtherPixel, class TUnaryOperation>
  static MutImage makeFromTransform(
      ImageView<TOtherPixel> view, TUnaryOperation const& unary_op) {
    MutImage mut_image(view.imageSize());
    mut_image.transformFrom(view, unary_op);
    return mut_image;
  }

  /// Creates new MutImage given two views and binary transform function.
  ///
  /// mut_image(u, v) = binary_op(lhs(u, v), rhs(u, v));
  template <class TLhsPixel, class TRhsPixel, class TBinaryOperation>
  static MutImage makeFromTransform(
      ImageView<TLhsPixel> lhs,
      ImageView<TRhsPixel> rhs,
      TBinaryOperation const& binary_op) {
    MutImage mut_image(lhs.imageSize());
    mut_image.transformFrom(lhs, rhs, binary_op);
    return mut_image;
  }

  // Begin(Rule of 5):
  // https://en.cppreference.com/w/cpp/language/rule_of_three

  /// Destructor
  ~MutImage() { reset(); }

  /// Not copy constructable
  MutImage(MutImage<TPixel> const& other) = delete;

  /// Not copy assignable
  MutImage& operator=(MutImage const&) = delete;

  /// Move constructor - is cheap - no memory allocations.
  MutImage(MutImage&& img) noexcept
      : MutImageView<TPixel>(img.viewMut()), unique_(std::move(img.unique_)) {
    this->ptr_ = reinterpret_cast<TPixel*>(unique_.get());
    img.setViewToEmpty();
  }

  /// Move assignment
  MutImage& operator=(MutImage&& img) noexcept {
    reset();
    this->shape_ = img.shape_;
    this->unique_ = std::move(img.unique_);
    this->ptr_ = reinterpret_cast<TPixel*>(unique_.get());
    img.setViewToEmpty();
    return *this;
  }
  // End (Rule of 5)

  [[nodiscard]] MutImageView<TPixel> viewMut() const {
    return MutImageView<TPixel>(this->shape(), this->ptrMut());
  }

  /// Swaps img and this.
  void swap(MutImage& img) {
    std::swap(img.shape_, this->shape_);
    std::swap(img.ptr_, this->ptr_);
    std::swap(img.unique_, this->unique_);
  }

  /// Clears image.
  ///
  /// If image was not empty, memory deallocations will happen.
  void reset() {
    unique_.reset();
    this->setViewToEmpty();
  }

 protected:
  /// Leaks memory and returns deleter.
  Deleter leakAndReturnDeleter() {
    SOPHUS_ASSERT(!this->isEmpty());
    Deleter& del = *std::get_deleter<Deleter>(unique_);
    Deleter del_copy = del;
    del.image_deleter.reset();
    return del_copy;
  }

  /// MutImage has unique ownership.
  UniqueDataArea unique_;
};
}  // namespace sophus