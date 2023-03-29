// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/enum.h"
#include "sophus/concepts/lie_group.h"
#include "sophus/lie/impl/semi_direct_product.h"

namespace sophus {
namespace lie {

template <class TDerived, class TImpl>
class Group {
 public:
  using Impl = TImpl;
  using Scalar = typename Impl::Scalar;

  static bool constexpr kIsOriginPreserving = Impl::kIsOriginPreserving;
  static bool constexpr kIsAxisDirectionPreserving =
      Impl::kIsAxisDirectionPreserving;
  static bool constexpr kIsDirectionVectorPreserving =
      Impl::kIsDirectionVectorPreserving;
  static bool constexpr kIsShapePreserving = Impl::kIsShapePreserving;
  static bool constexpr kIisSizePreserving = Impl::kIisSizePreserving;
  static bool constexpr kIisParallelLinePreserving =
      Impl::kIisParallelLinePreserving;

  static int constexpr kDof = Impl::kDof;
  static int constexpr kNumParams = Impl::kNumParams;
  static int constexpr kPointDim = Impl::kPointDim;
  static int constexpr kAmbientDim = Impl::kAmbientDim;

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;

  // constructors and factories

  Group() : params_(Impl::identityParams()) {}

  Group(Group const&) = default;
  auto operator=(Group const&) -> Group& = default;

  static auto fromParams(Params const& params) -> TDerived {
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

  static auto exp(Tangent const& tangent) -> TDerived {
    return TDerived::fromParamsUnchecked(Impl::exp(tangent));
  }

  [[nodiscard]] auto log() const -> Tangent { return Impl::log(this->params_); }

  static auto hat(Tangent const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Impl::hat(tangent);
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Tangent {
    return Impl::vee(mat);
  }

  // group operations

  auto operator*(TDerived const& rhs) const -> TDerived {
    return TDerived::fromParamsUnchecked(
        Impl::multiplication(this->params_, rhs.params_));
  }

  auto operator*=(TDerived const& rhs) -> TDerived& {
    *this = *this * rhs;
    return self();
  }

  [[nodiscard]] auto inverse() const -> TDerived {
    return TDerived::fromParams(Impl::inverse(this->params_));
  }

  // Group actions

  auto operator*(Point const& point) const -> Point {
    return Impl::action(this->params_, point);
  }

  static auto toAmbient(Point const& point) { return Impl::toAmbient(point); }

  auto operator*(UnitVector<Scalar, kPointDim> const& direction_vector) const
      -> UnitVector<Scalar, kPointDim> {
    return Impl::action(params_, direction_vector);
  }

  [[nodiscard]] auto adj() const -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Impl::adj(this->params_);
  }

  // left addition also called "left translation" in the literature
  [[nodiscard]] auto leftPlus(Tangent const& tangent) const -> TDerived {
    return this->exp(tangent) * self();
  }

  // right addition also called "right translation" in the literature
  [[nodiscard]] auto rightPlus(Tangent const& tangent) const -> TDerived {
    return self() * exp(tangent);
  }

  [[nodiscard]] auto leftMinus(TDerived const& other) const -> Tangent {
    return (self() * other.inverse()).log();
  }

  [[nodiscard]] auto rightMinus(TDerived const& other) const -> Tangent {
    return (other.inverse() * self()).log();
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

  static auto ad(Tangent const& tangent) -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Impl::ad(tangent);
  }

  // static auto dxExpX(Tangent const& tangent)
  //     -> Eigen::Matrix<Scalar, kNumParams, kDof> {
  //   return Impl::dxExpX(tangent);
  // }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Impl::dxExpXAt0();
  }

  static auto dxExpXTimesPointAt0(Point const& point)
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

  static auto tangentExamples() -> std::vector<Tangent> {
    return Impl::tangentExamples();
  }

  static auto paramsExamples() -> std::vector<Params> {
    return Impl::paramsExamples();
  }

  static auto elementExamples() -> std::vector<TDerived> {
    std::vector<TDerived> out;
    for (auto const& params : TDerived::paramsExamples()) {
      out.push_back(TDerived::fromParams(params));
    }
    return out;
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    return Impl::invalidParamsExamples();
  }

  // getters and setters

  [[nodiscard]] auto params() const -> Params const& { return this->params_; }

  [[nodiscard]] auto ptr() const { return this->params_.data(); }

  [[nodiscard]] auto unsafeMutPtr() { return this->params_.data(); }

  void setParams(Params const& params) {
    // Hack to get unexpected error message on failure.
    auto maybe_valid = Impl::areParamsValid(params);
    SOPHUS_UNWRAP(maybe_valid);
    this->params_ = params;
  }

 protected:
  explicit Group(UninitTag /*unused*/) {}

  auto self() -> TDerived& { return static_cast<TDerived&>(*this); }

  auto self() const -> TDerived const& {
    return static_cast<TDerived const&>(*this);
  }

  static auto fromParamsUnchecked(Params const& params) -> TDerived {
    TDerived g(UninitTag{});
    g.setParamsUnchecked(params);
    return g;
  }

  void setParamsUnchecked(Params const& params) { this->params_ = params; }

  Params params_;
};
}  // namespace lie

}  // namespace sophus
