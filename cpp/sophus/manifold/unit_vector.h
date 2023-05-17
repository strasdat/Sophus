// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/concepts/manifold.h"

#include <Eigen/Dense>

namespace sophus {

// Forward declarations
template <class TScalar, int kN>
class UnitVector;

// Convenience typedefs
template <class TScalar>
using UnitVector3 = UnitVector<TScalar, 3>;
template <class TScalar>
using UnitVector2 = UnitVector<TScalar, 2>;

using UnitVector2F64 = UnitVector2<double>;
using UnitVector3F64 = UnitVector3<double>;

namespace linalg {
template <class TScalar, int kDim>
class UnitVectorImpl {
 public:
  using Scalar = TScalar;

  static int constexpr kDof = kDim - 1;
  static int constexpr kNumParams = kDim;

  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Tangent = Eigen::Vector<Scalar, kDof>;

  static auto areParamsValid(Params const& unit_vector)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar>;
    const Scalar squared_norm = unit_vector.squaredNorm();
    using std::abs;
    if (!(abs(squared_norm - 1.0) <= kThr)) {
      return SOPHUS_UNEXPECTED(
          "unit vector ({}) is not of unit length.\n"
          "Squared norm: {}, thr: {}",
          unit_vector.transpose(),
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  static auto oplus(Params const& params, Tangent const& delta) -> Params {
    return matRx(params) * exp(delta);
  }

  static auto ominus(Params const& lhs_params, Params const& rhs_params)
      -> Tangent {
    return log((matRx(lhs_params).transpose() * rhs_params).eval());
  }

  static auto paramsExamples() -> std::vector<Params> {
    return std::vector<Params>(
        {Params::UnitX(), Params::UnitY(), -Params::UnitX(), -Params::UnitY()});
  }
  static auto invalidParamsExamples() -> std::vector<Params> {
    return std::vector<Params>(
        {Params::Zero(), Params::Ones(), 2.0 * Params::UnitX()});
  }

  static auto tangentExamples() -> std::vector<Tangent> {
    return std::vector<Tangent>({
        Tangent::Zero(),
        0.01 * Tangent::UnitX(),
        0.001 * Tangent::Ones(),
    });
  }

 private:
  static auto matRx(Params const& params)
      -> Eigen::Matrix<Scalar, kNumParams, kNumParams> {
    static Eigen::Vector<TScalar, kDim> const kUnitX =
        Eigen::Vector<TScalar, kDim>::UnitX();
    if ((kUnitX - params).squaredNorm() < kEpsilon<Scalar>) {
      return Eigen::Matrix<Scalar, kNumParams, kNumParams>::Identity();
    }
    Params v = params - kUnitX;
    return Eigen::Matrix<Scalar, kNumParams, kNumParams>::Identity() -
           2.0 * (v * v.transpose()) / v.squaredNorm();
  }

  static auto exp(Tangent const& delta) -> Params {
    using std::cos;
    Params params;
    Scalar theta = delta.norm();
    params[0] = cos(theta);
    params.template tail<kDof>() = sinc(theta) * delta;
    return params;
  }

  static auto log(Params const& params) -> Tangent {
    using std::atan2;

    static Tangent const kUnitX = Tangent::UnitX();
    Scalar x = params[0];
    Tangent tail = params.template tail<kDof>();
    Scalar theta = tail.norm();

    if (abs(theta) < kEpsilon<Scalar>) {
      return atan2(Scalar(0.0), x) * kUnitX;
    }

    return (1.0 / theta) * atan2(theta, x) * tail;
  }

  static auto sinc(Scalar x) -> Scalar {
    using std::abs;
    using std::sin;
    if (abs(x) < kEpsilon<Scalar>) {
      return 1.0 - (1.0 / 6.0) * (x * x);
    }
    return sin(x) / x;
  }
};

}  // namespace linalg

template <class TScalar, int kN>
class UnitVector : public linalg::UnitVectorImpl<TScalar, kN> {
 public:
  using Scalar = TScalar;
  using Impl = linalg::UnitVectorImpl<Scalar, kN>;
  static_assert(concepts::ManifoldImpl<Impl>);

  static int constexpr kDof = kN - 1;
  static int constexpr kNumParams = kN;
  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;

  static auto tryFromUnitVector(Eigen::Matrix<TScalar, kN, 1> const& v)
      -> Expected<UnitVector> {
    SOPHUS_TRY(auto, maybe_valid, Impl::areParamsValid(v));
    UnitVector unit_vector;
    unit_vector.vector_ = v;
    return unit_vector;
  }

  static auto fromParams(Eigen::Matrix<TScalar, kN, 1> const& v) -> UnitVector {
    Expected<UnitVector> e_vec = tryFromUnitVector(v);
    if (!e_vec.has_value()) {
      SOPHUS_PANIC("{}", e_vec.error());
    }
    return e_vec.value();
  }

  // Precondition: v must be of unit length.
  static auto fromUnitVector(Eigen::Matrix<TScalar, kN, 1> const& v)
      -> UnitVector {
    return fromParams(v);
  }

  static auto fromVectorAndNormalize(Eigen::Matrix<TScalar, kN, 1> const& v)
      -> UnitVector {
    return fromUnitVector(v.normalized());
  }

  auto oplus(Tangent const& delta) const -> UnitVector {
    UnitVector v;
    v.vector_ = Impl::oplus(vector_, delta);
    return v;
  }

  auto ominus(UnitVector const& rhs_params) const -> Tangent {
    return Impl::ominus(vector_, rhs_params.vector_);
  }

  void setParams(Eigen::Matrix<TScalar, kN, 1> const& v) const {
    SOPHUS_ASSERT(Impl::areParamsValid(v));
    vector_.params = v;
  }

  [[nodiscard]] auto params() const -> Eigen::Matrix<TScalar, kN, 1> const& {
    return vector_;
  }

  [[nodiscard]] auto vector() const -> Eigen::Matrix<TScalar, kN, 1> const& {
    return vector_;
  }

  [[nodiscard]] auto unsafeMutPtr() { return this->vector_.data(); }
  [[nodiscard]] auto ptr() const { return this->vector_.data(); }

  UnitVector(UnitVector const&) = default;
  auto operator=(UnitVector const&) -> UnitVector& = default;

  template <concepts::Range TSequenceContainer>
  static auto average(TSequenceContainer const& range) -> UnitVector {
    size_t const len = std::distance(std::begin(range), std::end(range));
    SOPHUS_ASSERT_GE(len, 0);

    Params params = Params::Zero();
    for (auto const& m : range) {
      params += m.params();
    }
    return fromVectorAndNormalize(params / len);
  }

 private:
  UnitVector() {}

  // Class invariant: v_ is of unit length.
  Eigen::Matrix<TScalar, kN, 1> vector_;
};

static_assert(concepts::Manifold<UnitVector<double, 3>>);

}  // namespace sophus
