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

#include "sophus/image/image_view.h"

#include <farm_ng/core/enum/enum.h>
#include <farm_ng/core/logging/logger.h>

#include <optional>

namespace sophus {

// Types are largely inspired / derived from Pangolin.

template <class PixelT, template <typename> class AllocatorT>
class Image;

/// A image with write access to pixels and exclusive ownership. There is no
/// copy constr / copy assignment, but move constr / assignment.
///
/// Content from a MutImage can be moved into an Image.
///
/// Type is nullable. In that case `this->isEmpty()` is true.
///
/// Similar to Pangolin::ManagedImage.
template <
    class PixelT,
    template <typename> class AllocatorT = Eigen::aligned_allocator>
class MutImage : public MutImageView<PixelT> {
 public:
  struct TypedDeleterImpl {
    TypedDeleterImpl(size_t num_bytes) : num_bytes(num_bytes) {}

    void operator()(PixelT* p) const {
      if (p != nullptr) {
        AllocatorT<PixelT>().deallocate(p, num_bytes / sizeof(PixelT));
      }
    }

    size_t num_bytes;
  };

  /// Deleter for MutImage.
  struct Deleter {
    Deleter(TypedDeleterImpl image_deleter) : image_deleter(image_deleter) {}

    void operator()(uint8_t* p) const {
      if (image_deleter) {
        (*image_deleter)(reinterpret_cast<PixelT*>(p));
      }
    }

    std::optional<TypedDeleterImpl> image_deleter;
  };

  template <class T2T, template <typename> class Allocator2T>
  friend class Image;

  /// Constructs empty image.
  MutImage() = default;

  /// Creates new image with given shape.
  ///
  /// If shape is not empty, memory allocation will happen.
  explicit MutImage(ImageShape shape) : MutImageView<PixelT>(shape, nullptr) {
    if (shape.sizeBytes() != 0u) {
      FARM_CHECK_EQ(shape.sizeBytes() % sizeof(PixelT), 0);
      this->shared_.reset(
          (uint8_t*)(AllocatorT<PixelT>().allocate(
              shape.sizeBytes() / sizeof(PixelT))),
          Deleter(TypedDeleterImpl(shape.sizeBytes())));
    }
    this->ptr_ = reinterpret_cast<PixelT*>(this->shared_.get());
  }

  /// Creates new contiguous image with given size.
  ///
  /// If shape is not empty, memory allocation will happen.
  explicit MutImage(sophus::ImageSize size)
      : MutImage(ImageShape::makeFromSize<PixelT>(size)) {}

  /// Creates contiguous copy from view.
  ///
  /// If view is not empty, memory allocation will happen.
  [[nodiscard]] static MutImage makeCopyFrom(const ImageView<PixelT>& view) {
    MutImage image(view.imageSize());
    image.copyDataFrom(view);
    return image;
  }

  /// Creates new MutImage given view and unary transform function.
  ///
  /// mut_image(u, v) = unary_op(view(u, v));
  template <class OtherPixelT, class UnaryOperationT>
  static MutImage makeFromTransform(
      ImageView<OtherPixelT> view, const UnaryOperationT& unary_op) {
    MutImage mut_image(view.imageSize());
    mut_image.transformFrom(view, unary_op);
    return mut_image;
  }

  /// Creates new MutImage given two views and binary transform function.
  ///
  /// mut_image(u, v) = binary_op(lhs(u, v), rhs(u, v));
  template <class LhsPixelT, class RhsPixelT, class BinaryOperationT>
  static MutImage makeFromTransform(
      ImageView<LhsPixelT> lhs,
      ImageView<RhsPixelT> rhs,
      const BinaryOperationT& binary_op) {
    MutImage mut_image(lhs.imageSize());
    mut_image.transformFrom(lhs, rhs, binary_op);
    return mut_image;
  }

  // Begin(Rule of 5):
  // https://en.cppreference.com/w/cpp/language/rule_of_three

  /// Destructor
  ~MutImage() { reset(); }

  /// Not copy constructable
  MutImage(const MutImage<PixelT>& other) = delete;

  /// Not copy assignable
  MutImage& operator=(const MutImage&) = delete;

  /// Move constructor - is cheap - no memory allocations.
  MutImage(MutImage&& img) noexcept
      : MutImageView<PixelT>(img.mutView()), shared_(std::move(img.shared_)) {
    this->ptr_ = reinterpret_cast<PixelT*>(shared_.get());
    img.setViewToEmpty();
  }

  /// Move assignment
  MutImage& operator=(MutImage&& img) noexcept {
    reset();
    this->shape_ = img.shape_;
    this->shared_ = std::move(img.shared_);
    this->ptr_ = reinterpret_cast<PixelT*>(shared_.get());
    img.setViewToEmpty();
    return *this;
  }
  // End (Rule of 5)

  [[nodiscard]] MutImageView<PixelT> mutView() const {
    return MutImageView<PixelT>(this->shape(), this->mutPtr());
  }

  /// Swaps img and this.
  void swap(MutImage& img) {
    std::swap(img.shape_, this->shape_);
    std::swap(img.ptr_, this->ptr_);
    std::swap(img.shared_, this->shared_);
  }

  /// Clears image.
  ///
  /// If image was not empty, memory deallocations will happen.
  void reset() {
    shared_.reset();
    this->setViewToEmpty();
  }

 protected:
  /// Leaks memory and returns deleter.
  Deleter leakAndReturnDeleter() {
    FARM_CHECK(!this->isEmpty());
    Deleter& del = *std::get_deleter<Deleter>(shared_);
    Deleter del_copy = del;
    del.image_deleter.reset();
    return del_copy;
  }

  /// MutImage has unique ownership, and hence behaves like a unique_ptr. As an
  /// implementation detail, we use a shared_ptr here, so it will be easy to
  /// support moving a Image with unique ownership at runtime into a
  /// MutImage.
  /// Class invariant: shared_.use_count() == 0 || shared_.use_count() == 1.
  std::shared_ptr<uint8_t> shared_;  // NOLINT
};

template <
    template <typename>
    class PredicateT,
    template <typename>
    class AllocatorT>
class RuntimeImage;

/// Image read-only access to pixels and shared ownership, hence cheap to copy.
/// Type is nullable.
///
/// Image has close interop with RuntimeImage (see below).
template <
    class PixelT,
    template <typename> class AllocatorT = Eigen::aligned_allocator>
class Image : public ImageView<PixelT> {
 public:
  /// Constructs empty image.
  Image() = default;

  /// Moves MutImage into this.
  /// By design not "explicit".
  Image(MutImage<PixelT, AllocatorT>&& image) noexcept
      : ImageView<PixelT>(image.view()) {
    if (!image.isEmpty()) {
      this->shared_ = std::move(image.shared_);
      this->ptr_ = reinterpret_cast<PixelT*>(shared_.get());
      image.setViewToEmpty();
    }
  }

  /// Creates contiguous copy from view.
  ///
  /// If view is not empty, memory allocation will happen.
  [[nodiscard]] static Image makeCopyFrom(const ImageView<PixelT>& view) {
    return Image(MutImage<PixelT>::makeCopyFrom(view));
  }

  /// Creates new Image given view and unary transform function.
  ///
  /// image(u, v) = unary_op(view(u, v));
  template <class OtherPixelT, class UnaryOperationT>
  static Image makeFromTransform(
      ImageView<OtherPixelT> view, const UnaryOperationT& unary_op) {
    return MutImage<PixelT>::makeFromTransform(view, unary_op);
  }

  /// Creates new Image given two views and binary transform function.
  ///
  /// image(u, v) = binary_op(lhs(u, v), rhs(u, v));
  template <class LhsPixelT, class RhsPixelT, class BinaryOperationT>
  static Image makeFromTransform(
      ImageView<LhsPixelT> lhs,
      ImageView<RhsPixelT> rhs,
      const BinaryOperationT& binary_op) {
    return MutImage<PixelT>::makeFromTransform(lhs, rhs, binary_op);
  }

  [[nodiscard]] size_t useCount() const { return shared_.use_count(); }

  /// Sets Image instance to empty. Reduced use count by one.
  ///
  /// If use count goes to zero, deallocation happens.
  ///
  /// No-op if empty.
  void reset() {
    shared_.reset();
    this->setViewToEmpty();
  }

 private:
  template <class T2T, template <typename> class Allocator2T>
  friend class MutImage;

  template <
      template <typename>
      class PredicateT,
      template <typename>
      class Allocator2T>
  friend class RuntimeImage;

  explicit Image(ImageView<PixelT> view) : ImageView<PixelT>(view) {}

  Image(ImageView<PixelT> view, const std::shared_ptr<uint8_t>& shared)
      : ImageView<PixelT>(view), shared_(shared) {}

  std::shared_ptr<uint8_t> shared_;
};

}  // namespace sophus
