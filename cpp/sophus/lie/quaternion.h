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
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using Params = typename Eigen::internal::traits<TDerived>::ParamsType;

  /// Returns 3x3 matrix representation of the quaternion.
  ///
  /// For SO(3), the matrix representation is an orthogonal matrix ``R`` with
  /// ``det(R)=1``, thus the so-called "rotation matrix".
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix3<Scalar> matrix() const {
    Eigen::Matrix3<Scalar> mat;

    Scalar const vx_sq = this->imag().x() * this->imag().x();
    Scalar const vy_sq = this->imag().y() * this->imag().y();
    Scalar const vz_sq = this->imag().z() * this->imag().z();
    Scalar const w_sq = this->real() * this->real();
    Scalar const two_vx = Scalar(2) * this->imag().x();
    Scalar const two_vy = Scalar(2) * this->imag().y();
    Scalar const two_vz = Scalar(2) * this->imag().z();
    Scalar const two_vx_vy = two_vx * this->imag().y();
    Scalar const two_vx_vz = two_vx * this->imag().z();
    Scalar const two_vx_w = two_vx * this->real();
    Scalar const two_vy_vz = two_vy * this->imag().z();
    Scalar const two_vy_w = two_vy * this->real();
    Scalar const two_vz_w = two_vz * this->real();

    mat(0, 0) = vx_sq - vy_sq - vz_sq + w_sq;
    mat(1, 0) = two_vx_vy + two_vz_w;
    mat(2, 0) = two_vx_vz - two_vy_w;

    mat(0, 1) = two_vx_vy - two_vz_w;
    mat(1, 1) = -vx_sq + vy_sq - vz_sq + w_sq;
    mat(2, 1) = two_vx_w + two_vy_vz;

    mat(0, 2) = two_vx_vz + two_vy_w;
    mat(1, 2) = -two_vx_w + two_vy_vz;
    mat(2, 2) = -vx_sq - vy_sq + vz_sq + w_sq;
    return mat;
  }

  Scalar squaredNorm() const {
    return params().squaredNorm();
  }

  /// Accessor of params.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Map<Eigen::Vector3<Scalar> const> imag() const {
    return Eigen::Map<Eigen::Vector3<Scalar>const>(params().data());
  }

  /// Accessor of params.
  ///
  SOPHUS_FUNC [[nodiscard]] Params const& params() const {
    return static_cast<TDerived const*>(this)->params();
  }

  SOPHUS_FUNC [[nodiscard]] Scalar const& real() const { return params()[3]; }

  SOPHUS_FUNC [[nodiscard]] Scalar& real() { return mutParams()[3]; }

  /// Mutator of params
  ///
  SOPHUS_FUNC
  Params& mutParams() { return static_cast<TDerived*>(this)->mutParams(); }
};

/// Quaternion using  default storage; derived from So2Base.
template <class TScalar>
class Quaternion : public QuaternionBase<Quaternion<TScalar>> {
 public:
  using Scalar = TScalar;

  SOPHUS_FUNC static Quaternion fromRealAndImag(
      Scalar real, Eigen::Vector3<Scalar> const& imag_vec) {
    Quaternion quat;
    quat.params_.template head<3>() = imag_vec;
    quat.params_[3] = real;
  }

 private:
  Quaternion() = default;

  Eigen::Matrix<Scalar, 4, 1> params_;  // NOLINT
};

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``So2``; derived from So2Base.
///
/// Allows us to wrap So2 objects around POD array.
template <class TScalar>
class Map<sophus::Quaternion<TScalar>>
    : public sophus::QuaternionBase<Map<sophus::Quaternion<TScalar>>> {
 public:
  using Scalar = TScalar;

  SOPHUS_FUNC explicit Map(Scalar* coeffs) : params_(coeffs) {}

 protected:
  Map<Eigen::Matrix<Scalar, 4, 1>> params_;  // NOLINT
};

/// Specialization of Eigen::Map for ``So2 const``; derived from So2Base.
///
/// Allows us to wrap So2 objects around POD array (e.g. external c style
/// complex number / tuple).
template <class TScalar>
class Map<sophus::Quaternion<TScalar> const>
    : public sophus::QuaternionBase<Map<sophus::Quaternion<TScalar> const>> {
 public:
  using Scalar = TScalar;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs) : params_(coeffs) {}

 protected:
  Map<Eigen::Matrix<Scalar, 4, 1> const> params_;  // NOLINT
};

}  // namespace Eigen
