// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "eigen_concepts.h"

#include <Eigen/Core>

#include <limits>

namespace sophus {

// Ideally we could use Eigen's own trait system but it appears to be missing
// some important traits such as lowest(), min() and max().
//
// template<typename T>
// using MultiDimLimits = Eigen::NumTraits<T>;

// Probably frowned upon, but we'll inherit from the std::library traits
// for ordinary scalars etc.
template <class T>
struct MultiDimLimits : public std::numeric_limits<T> {};

template <EigenDenseType TT>
class MultiDimLimits<TT> {
 public:
  using TScalar = typename TT::Scalar;
  static int constexpr kRows = TT::RowsAtCompileTime;
  static int constexpr kCols = TT::ColsAtCompileTime;

  static bool constexpr is_specialized = true;
  static bool constexpr has_infinity =
      std::numeric_limits<TScalar>::has_infinity;
  static bool constexpr has_quiet_NaN =
      std::numeric_limits<TScalar>::has_quiet_NaN;
  static bool constexpr has_signaling_NaN =
      std::numeric_limits<TScalar>::has_signaling_NaN;
  /// ... plus a bunch more if we need them

  static TT lowest() {
    return TT::Constant(std::numeric_limits<TScalar>::lowest());
  };
  static TT min() { return TT::Constant(std::numeric_limits<TScalar>::min()); };
  static TT max() { return TT::Constant(std::numeric_limits<TScalar>::max()); };
  static TT epsilon() {
    return TT::Constant(std::numeric_limits<TScalar>::epsilon());
  };
  /// ... plus a bunch more if we need them
};

}  // namespace sophus
