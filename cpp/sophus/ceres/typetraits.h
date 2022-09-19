// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/core/common.h"

#include <ceres/ceres.h>

#include <type_traits>

namespace sophus {

template <class TT, std::size_t = sizeof(TT)>
constexpr std::true_type complete(TT*);
constexpr std::false_type complete(...);

template <class TT>
using IsSpecialized = decltype(complete(std::declval<TT*>()));

/// Type trait used to distinguish mappable vector types from scalars
///
/// We use this class to distinguish Eigen::Vector<Scalar, kMatrixDim> from
/// Scalar types in LieGroup<T>::Tangent
///
/// Primary use is mapping LieGroup::Tangent over raw data, with 2 options:
///  - LieGroup::Tangent is "scalar" (for So2), then we just dereference pointer
///  - LieGroup::Tangent is Eigen::Vector<...>, then we need to use Eigen::Map
///
/// Specialization of Eigen::internal::traits<T> for T is crucial for
/// for constructing Eigen::Map<T>, thus we use that property for distinguishing
/// between those two options.
/// At this moment there seem to be no option to check this using only
/// "external" API of Eigen
template <class TT>
using IsMappable = IsSpecialized<Eigen::internal::traits<std::decay_t<TT>>>;

template <class TT>
constexpr bool kIsMappableV = IsMappable<TT>::value;

/// Helper for mapping tangent vectors (scalars) over pointers to data
template <typename TT, typename ET = void>
struct Mapper {
  using Scalar = TT;
  using Map = Scalar&;
  using ConstMap = const Scalar&;

  static Map map(Scalar* ptr) noexcept { return *ptr; }
  static ConstMap map(const Scalar* ptr) noexcept { return *ptr; }
};

template <typename TT>
struct Mapper<TT, typename std::enable_if<kIsMappableV<TT>>::type> {
  using Scalar = typename TT::Scalar;
  using Map = Eigen::Map<TT>;
  using ConstMap = Eigen::Map<const TT>;

  static Map map(Scalar* ptr) noexcept { return Map(ptr); }
  static ConstMap map(const Scalar* ptr) noexcept { return ConstMap(ptr); }
};

}  // namespace sophus
