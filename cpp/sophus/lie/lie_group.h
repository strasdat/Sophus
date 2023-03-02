// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/lie_group.h"
#include "sophus/lie/impl/semi_direct_product.h"

namespace sophus {
namespace lie {

template <class TDerived, class TImpl>
class Group {
 public:
  using Impl = TImpl;
  using Scalar = typename Impl::Scalar;
  static int constexpr kDof = Impl::kDof;
  static int constexpr kNumParams = Impl::kNumParams;
  static int constexpr kPointDim = Impl::kPointDim;
  static int constexpr kAmbientDim = Impl::kAmbientDim;

  // constructors and factories

  Group() : params_(Impl::identityParams()) {}

  Group(Group const&) = default;
  auto operator=(Group const&) -> Group& = default;

  static auto fromParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> TDerived {
    TDerived g(UninitTag{});
    g.setParams(params);
    return g;
  }

  static auto identity() -> TDerived {
    TDerived g(UninitTag{});
    g.setParams(Impl::identityParams);
    return g;
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& tangent) -> TDerived {
    return TDerived::fromParamsUnchecked(Impl::exp(tangent));
  }

  [[nodiscard]] auto log() const -> Eigen::Vector<Scalar, kDof> {
    return Impl::log(this->params_);
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Impl::hat(tangent);
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Vector<Scalar, kDof> {
    return Impl::vee(mat);
  }

  // group operations

  auto operator*(TDerived const& rhs) const -> TDerived {
    return TDerived::fromParamsUnchecked(
        Impl::multiplication(this->params_, rhs.params_));
  }

  [[nodiscard]] auto inverse() const -> TDerived {
    return TDerived::fromParams(Impl::inverse(this->params_));
  }

  // Point actions

  auto operator*(Eigen::Vector<Scalar, kPointDim> const& point) const
      -> Eigen::Vector<Scalar, kPointDim> {
    return Impl::action(this->params_, point);
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point) {
    return Impl::toAmbient(point);
  }

  auto operator*(UnitVector<Scalar, kPointDim> const& direction_vector)
      -> UnitVector<Scalar, kPointDim> {
    return Impl::action(params_, direction_vector);
  }

  // Matrices

  [[nodiscard]] auto compactMatrix() const
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return Impl::compactMatrix(this->params_);
  }

  [[nodiscard]] auto matrix() const
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Impl::matrix(this->params_);
  }

  // derivatives

  [[nodiscard]] auto adj() const -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Impl::adj(this->params_);
  }

  // static auto dxExpX(Eigen::Vector<Scalar, kDof> const& tangent)
  //     -> Eigen::Matrix<Scalar, kNumParams, kDof> {
  //   return Impl::dxExpX(tangent);
  // }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Impl::dxExpXAt0();
  }

  static auto dxExpXTimesPointAt0(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Impl::dxExpXTimesPointAt0(point);
  }

  [[nodiscard]] auto dxThisMulExpXAt0() const
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Impl::dxThisMulExpXAt0(this->params_);
  }

  [[nodiscard]] auto dxLogThisInvTimesXAtThis() const
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    return Impl::dxLogThisInvTimesXAtThis(this->params_);
  }

  // for tests

  static auto tangentExamples() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return Impl::tangentExamples();
  }

  static auto paramsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return Impl::paramsExamples();
  }

  static auto elementExamples() -> std::vector<TDerived> {
    std::vector<TDerived> out;
    for (auto const& params : TDerived::paramsExamples()) {
      out.push_back(TDerived::fromParams(params));
    }
    return out;
  }

  static auto invalidParamsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return Impl::invalidParamsExamples();
  }

  // getters and setters

  [[nodiscard]] auto params() const
      -> Eigen::Vector<Scalar, kNumParams> const& {
    return this->params_;
  }

  [[nodiscard]] auto ptr() const { return this->params_.data(); }

  [[nodiscard]] auto unsafeMutPtr() { return this->params_.data(); }

  void setParams(Eigen::Vector<Scalar, kNumParams> const& params) {
    // Hack to get unexpected error message on failure.
    auto maybe_valid = Impl::areParamsValid(params);
    SOPHUS_UNWRAP(maybe_valid);
    this->params_ = params;
  }

 protected:
  explicit Group(UninitTag /*unused*/) {}

  static auto fromParamsUnchecked(
      Eigen::Vector<Scalar, kNumParams> const& params) -> TDerived {
    TDerived g(UninitTag{});
    g.setParamsUnchecked(params);
    return g;
  }

  void setParamsUnchecked(Eigen::Vector<Scalar, kNumParams> const& params) {
    this->params_ = params;
  }

  Eigen::Vector<Scalar, kNumParams> params_;
};

}  // namespace lie

}  // namespace sophus
