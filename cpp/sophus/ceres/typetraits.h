// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/common/common.h"

#include <ceres/ceres.h>

#include <type_traits>

namespace sophus {

template <class TScalar, std::size_t = sizeof(TScalar)>
constexpr std::true_type complete(TScalar*);
constexpr std::false_type complete(...);

template <class TScalar>
using IsSpecialized = decltype(complete(std::declval<TScalar*>()));

/// Type trait used to distinguish mappable vector types from scalars
///
/// We use this class to distinguish Eigen::Vector<Scalar, kMatrixDim> from
/// Scalar types in LieGroup<TScalar>::Tangent
///
/// Primary use is mapping LieGroup::Tangent over raw data, with 2 options:
///  - LieGroup::Tangent is "scalar" (for Rotation2), then we just dereference
///  pointer
///  - LieGroup::Tangent is Eigen::Vector<...>, then we need to use Eigen::Map
///
/// Specialization of Eigen::internal::traits<TScalar> for TScalar is crucial
/// for for constructing Eigen::Map<TScalar>, thus we use that property for
/// distinguishing between those two options. At this moment there seem to be no
/// option to check this using only "external" API of Eigen
template <class TScalar>
using IsMappable =
    IsSpecialized<Eigen::internal::traits<std::decay_t<TScalar>>>;

template <class TScalar>
bool constexpr kIsMappableV = IsMappable<TScalar>::value;

/// Helper for mapping tangent vectors (scalars) over pointers to data
template <class TScalar, typename TE = void>
struct Mapper {
  using Scalar = TScalar;
  using Map = Scalar&;
  using ConstMap = Scalar const&;

  static Map map(Scalar* ptr) noexcept { return *ptr; }
  static ConstMap map(Scalar const* ptr) noexcept { return *ptr; }
};

template <class TScalar>
struct Mapper<TScalar, typename std::enable_if<kIsMappableV<TScalar>>::type> {
  using Scalar = typename TScalar::Scalar;
  using Map = Eigen::Map<TScalar>;
  using ConstMap = Eigen::Map<const TScalar>;

  static Map map(Scalar* ptr) noexcept { return Map(ptr); }
  static ConstMap map(Scalar const* ptr) noexcept { return ConstMap(ptr); }
};

}  // namespace sophus
