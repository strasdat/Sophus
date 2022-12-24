// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Quaternion numbers.

#pragma once

#include "sophus/common/types.h"

namespace sophus {

template <class TScalar>
class Quaternion;
using QuaternionF64 = Quaternion<double>;
using QuaternionF32 = Quaternion<float>;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar>
struct traits<sophus::Quaternion<TScalar>> {
  using Scalar = TScalar;
  using ParamsType = Eigen::Matrix<Scalar, 4, 1>;
};

template <class TScalar>
struct traits<Map<sophus::Quaternion<TScalar>>>
    : traits<sophus::Quaternion<TScalar>> {
  using Scalar = TScalar;
  using ParamsType = Map<Eigen::Vector4<Scalar>>;
};

template <class TScalar>
struct traits<Map<sophus::Quaternion<TScalar> const>>
    : traits<sophus::Quaternion<TScalar> const> {
  using Scalar = TScalar;
  using ParamsType = Map<Eigen::Vector4<Scalar> const>;
};

}  // namespace internal
}  // namespace Eigen

namespace sophus {

template <class TDerived>
class QuaternionBase {
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using Params = typename Eigen::internal::traits<TDerived>::ParamsType;

  /// Accessor of params.
  ///
  SOPHUS_FUNC [[nodiscard]] Params const& params() const {
    return static_cast<TDerived*>(this)->params();
  }

  /// Mutator of params
  ///
  SOPHUS_FUNC
  Params& mutParams() { return static_cast<TDerived*>(this)->mutParams(); }
};

/// Quaternion using  default storage; derived from So2Base.
template <class TScalar>
class Quaternion : public QuaternionBase<Quaternion<TScalar>> {};

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``So2``; derived from So2Base.
///
/// Allows us to wrap So2 objects around POD array.
template <class TScalar>
class Map<sophus::Quaternion<TScalar>>
    : public sophus::QuaternionBase<Map<sophus::Quaternion<TScalar>>> {};

/// Specialization of Eigen::Map for ``So2 const``; derived from So2Base.
///
/// Allows us to wrap So2 objects around POD array (e.g. external c style
/// complex number / tuple).
template <class TScalar>
class Map<sophus::Quaternion<TScalar> const>
    : public sophus::QuaternionBase<Map<sophus::Quaternion<TScalar> const>> {};

}  // namespace Eigen
