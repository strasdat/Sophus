// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/common/common.h"
#include "sophus/concepts/division_ring.h"
#include "sophus/linalg/vector_space.h"

namespace sophus {

template <class TScalar>
class QuaternionImpl {
 public:
  using Scalar = TScalar;
  static int constexpr kNumParams = 4;
  static bool constexpr kIsCommutative = false;

  using Params = Eigen::Vector<Scalar, kNumParams>;

  template <class TCompatibleScalar>
  using ParamsReturn = Eigen::Vector<
      typename Eigen::ScalarBinaryOpTraits<Scalar, TCompatibleScalar>::
          ReturnType,
      4>;

  // factories

  static auto zero() -> Eigen::Vector<Scalar, 4> {
    return Eigen::Vector<Scalar, 4>::Zero();
  }

  static auto one() -> Eigen::Vector<Scalar, 4> {
    return Eigen::Vector<Scalar, 4>(0.0, 0.0, 0.0, 1.0);
  }

  static auto areParamsValid(Params const& /*unused*/)
      -> sophus::Expected<Success> {
    return sophus::Expected<Success>{};
  }

  static auto paramsExamples() -> std::vector<Params> {
    return pointExamples<Scalar, 4>();
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    return std::vector<Params>({});
  }

  template <class TCompatibleScalar>
  static auto multiplication(
      Eigen::Vector<Scalar, 4> const& lhs,
      Eigen::Vector<TCompatibleScalar, 4> const& rhs)
      -> ParamsReturn<TCompatibleScalar> {
    Eigen::Vector3<Scalar> lhs_ivec = lhs.template head<3>();
    Eigen::Vector3<TCompatibleScalar> rhs_ivec = rhs.template head<3>();

    ParamsReturn<TCompatibleScalar> out;
    out.w() = lhs.w() * rhs.w() - lhs_ivec.dot(rhs_ivec);
    out.template head<3>() =
        lhs.w() * rhs_ivec + rhs.w() * lhs_ivec + lhs_ivec.cross(rhs_ivec);
    return out;
  }

  template <class TCompatibleScalar>
  static auto addition(
      Eigen::Vector<Scalar, 4> const& a,
      Eigen::Vector<TCompatibleScalar, 4> const& b)
      -> ParamsReturn<TCompatibleScalar> {
    return a + b;
  }

  static auto conjugate(Eigen::Vector<Scalar, 4> const& a)
      -> Eigen::Vector<Scalar, 4> {
    return Eigen::Vector<Scalar, 4>(-a.x(), -a.y(), -a.z(), a.w());
  }

  static auto inverse(Eigen::Vector<Scalar, 4> const& q)
      -> Eigen::Vector<Scalar, 4> {
    return conjugate(q) / squaredNorm(q);
  }

  static auto norm(Eigen::Vector<Scalar, 4> const& q) -> Scalar {
    return q.norm();
  }

  static auto squaredNorm(Eigen::Vector<Scalar, 4> const& q) -> Scalar {
    return q.squaredNorm();
  }
};

template <class TScalar>
class Quaternion {
 public:
  using Scalar = TScalar;
  using Imag = Eigen::Vector<Scalar, 3>;
  using Impl = QuaternionImpl<Scalar>;
  static int constexpr kNumParams = 4;

  using Params = Eigen::Vector<Scalar, kNumParams>;

  template <class TCompatibleScalar>
  using QuaternionReturn = Quaternion<typename Eigen::ScalarBinaryOpTraits<
      Scalar,
      TCompatibleScalar>::ReturnType>;

  // constructors and factories

  Quaternion() : params_(Impl::zero()) {}

  Quaternion(Quaternion const&) = default;
  auto operator=(Quaternion const&) -> Quaternion& = default;

  static auto fromParams(Params const& params) -> Quaternion {
    Quaternion q(UninitTag{});
    q.setParams(params);
    return q;
  }

  static auto zero() -> Quaternion {
    return Quaternion::fromParams(Impl::zero());
  }

  static auto one() -> Quaternion {
    return Quaternion::fromParams(Impl::one());
  }

  [[nodiscard]] auto params() const -> Params const& { return params_; }

  void setParams(Params const& params) { params_ = params; }

  [[nodiscard]] auto real() const -> Scalar const& { return params_[3]; }
  [[nodiscard]] auto real() -> Scalar& { return params_[3]; }

  [[nodiscard]] auto imag() const { return params_.template head<3>(); }
  [[nodiscard]] auto imag() { return params_.template head<3>(); }

  template <class TCompatibleScalar>
  [[nodiscard]] auto operator*(Quaternion<TCompatibleScalar> const& other) const
      -> QuaternionReturn<TCompatibleScalar> {
    return QuaternionReturn<TCompatibleScalar>::fromParams(
        Impl::multiplication(this->params_, other.params()));
  }

  template <class TCompatibleScalar>
  [[nodiscard]] auto operator+(Quaternion<TCompatibleScalar> const& other) const
      -> QuaternionReturn<TCompatibleScalar> {
    return QuaternionReturn<TCompatibleScalar>::fromParams(
        Impl::addition(this->params_, other.params()));
  }

  [[nodiscard]] auto conjugate() const -> Quaternion {
    return Quaternion::fromParams(Impl::conjugate(this->params_));
  }

  [[nodiscard]] auto inverse() const -> Quaternion {
    return Quaternion::fromParams(Impl::inverse(this->params_));
  }

  [[nodiscard]] auto norm() const -> Scalar {
    return Impl::norm(this->params_);
  }

  [[nodiscard]] auto squaredNorm() const -> Scalar {
    return Impl::squaredNorm(this->params_);
  }

 private:
  Quaternion(UninitTag /*unused*/) {}
  Eigen::Vector4<Scalar> params_;
};

using QuaternionF64 = Quaternion<double>;

}  // namespace sophus
