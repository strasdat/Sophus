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
#include "sophus/image/mut_image.h"

#include <optional>

namespace sophus {

template <class TPredicate, template <class> class TAllocator>
class RuntimeImage;

/// Image read-only access to pixels and shared ownership, hence cheap to copy.
/// Type is nullable.
///
/// Image has close interop with RuntimeImage (see below).
template <
    class TPixel,
    template <class> class TAllocator = Eigen::aligned_allocator>
class Image : public ImageView<TPixel> {
 public:
  /// Constructs empty image.
  Image() = default;

  /// Moves MutImage into this.
  /// By design not "explicit".
  Image(MutImage<TPixel, TAllocator>&& image) noexcept
      : ImageView<TPixel>(image.view()) {
    if (!image.isEmpty()) {
      this->shared_ = std::move(image.shared_);
      this->ptr_ = reinterpret_cast<TPixel*>(shared_.get());
      image.setViewToEmpty();
    }
  }

  /// Creates contiguous copy from view.
  ///
  /// If view is not empty, memory allocation will happen.
  [[nodiscard]] static Image makeCopyFrom(ImageView<TPixel> const& view) {
    return Image(MutImage<TPixel>::makeCopyFrom(view));
  }

  /// Allocated and generates image from provided function taking u,v indices
  ///
  /// Memory allocation will happen.
  template <class TUVOperation>
  [[nodiscard]] static Image makeGenerative(
      ImageSize size, TUVOperation const& uv_op) {
    MutImage<TPixel> mut_image(size);
    mut_image.generate(uv_op);
    return mut_image;
  }

  /// Creates new Image given view and unary transform function.
  ///
  /// image(u, v) = unary_op(view(u, v));
  template <class TOtherPixel, class TUnaryOperation>
  static Image makeFromTransform(
      ImageView<TOtherPixel> view, TUnaryOperation const& unary_op) {
    return MutImage<TPixel>::makeFromTransform(view, unary_op);
  }

  /// Creates new Image given two views and binary transform function.
  ///
  /// image(u, v) = binary_op(lhs(u, v), rhs(u, v));
  template <class TLhsPixel, class TRhsPixel, class TBinaryOperation>
  static Image makeFromTransform(
      ImageView<TLhsPixel> lhs,
      ImageView<TRhsPixel> rhs,
      TBinaryOperation const& binary_op) {
    return MutImage<TPixel>::makeFromTransform(lhs, rhs, binary_op);
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
  template <class TT, template <class> class TAllocator2T>
  friend class MutImage;

  template <class TPredicate, template <class> class TAllocator2T>
  friend class RuntimeImage;

  explicit Image(ImageView<TPixel> view) : ImageView<TPixel>(view) {}

  Image(ImageView<TPixel> view, std::shared_ptr<uint8_t> const& shared)
      : ImageView<TPixel>(view), shared_(shared) {}

  std::shared_ptr<uint8_t> shared_;
};

}  // namespace sophus
