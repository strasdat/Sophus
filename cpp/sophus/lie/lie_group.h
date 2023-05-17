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
#include "sophus/lie/impl/translation_factor_group_product.h"

namespace sophus {
namespace lie {

template <
    template <class>
    class TGenericDerived,
    class TScalar,
    template <class>
    class TGenericImpl>
class Group {
 public:
  using Scalar = TScalar;
  using Impl = TGenericImpl<TScalar>;

  template <
      template <class>
      class TOtherGenericDerived,
      class TOtherScalar,
      template <class>
      class TOtherGenericImpl>
  friend class Group;

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

  template <class TCompatibleScalar>
  using ScalarReturn = typename Eigen::
      ScalarBinaryOpTraits<Scalar, TCompatibleScalar>::ReturnType;

  using Derived = TGenericDerived<Scalar>;

  template <class TCompatibleScalar>
  using DerivedReturn = TGenericDerived<ScalarReturn<TCompatibleScalar>>;

  template <class TCompatibleScalar>
  using PointReturn = Eigen::Vector<ScalarReturn<TCompatibleScalar>, kPointDim>;

  template <class TCompatibleScalar>
  using UnitVectorReturn =
      UnitVector<ScalarReturn<TCompatibleScalar>, kPointDim>;

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;

  // constructors and factories

  Group() : params_(Impl::identityParams()) {}

  Group(Group const&) = default;
  auto operator=(Group const&) -> Group& = default;

  static auto fromParams(Params const& params) -> Derived {
    Derived g(UninitTag{});
    g.setParams(params);
    return g;
  }

  static auto identity() -> Derived {
    Derived g(UninitTag{});
    g.setParams(Impl::identityParams);
    return g;
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& tangent) -> Derived {
    return Derived::fromParamsUnchecked(Impl::exp(tangent));
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

  auto hasShortestPathAmbiguity() -> bool {
    return Impl::hasShortestPathAmbiguity(this->params_);
  }

  // group operations

  template <class TCompatibleScalar>
  auto operator*(TGenericDerived<TCompatibleScalar> const& rhs) const
      -> DerivedReturn<TCompatibleScalar> {
    auto params = Impl::multiplication(this->params_, rhs.params());
    return DerivedReturn<TCompatibleScalar>::fromParamsUnchecked(params);
  }

  auto operator*=(Derived const& rhs) -> Derived& {
    *this = *this * rhs;
    return self();
  }

  [[nodiscard]] auto inverse() const -> Derived {
    return Derived::fromParams(Impl::inverse(this->params_));
  }

  // Group actions
  template <class TMatrixDerived>
  auto operator*(Eigen::MatrixBase<TMatrixDerived> const& point) const
      -> PointReturn<typename TMatrixDerived::Scalar> {
    return Impl::action(this->params_, point.eval());
  }

  template <class TCompatibleScalar>
  auto operator*(
      UnitVector<TCompatibleScalar, kPointDim> const& direction_vector) const
      -> UnitVectorReturn<TCompatibleScalar> {
    return Impl::action(params_, direction_vector);
  }

  static auto toAmbient(Point const& point) { return Impl::toAmbient(point); }

  [[nodiscard]] auto adj() const -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Impl::adj(this->params_);
  }

  // left addition also called "left translation" in the literature
  [[nodiscard]] auto leftPlus(Tangent const& tangent) const -> Derived {
    return this->exp(tangent) * self();
  }

  // right addition also called "right translation" in the literature
  [[nodiscard]] auto rightPlus(Tangent const& tangent) const -> Derived {
    return self() * exp(tangent);
  }

  [[nodiscard]] auto leftMinus(Derived const& other) const -> Tangent {
    return (self() * other.inverse()).log();
  }

  [[nodiscard]] auto rightMinus(Derived const& other) const -> Tangent {
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

  static auto elementExamples() -> std::vector<Derived> {
    std::vector<Derived> out;
    for (auto const& params : Derived::paramsExamples()) {
      out.push_back(Derived::fromParams(params));
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

  auto self() -> Derived& { return static_cast<Derived&>(*this); }

  auto self() const -> Derived const& {
    return static_cast<Derived const&>(*this);
  }

  static auto fromParamsUnchecked(Params const& params) -> Derived {
    Derived g(UninitTag{});
    g.setParamsUnchecked(params);
    return g;
  }

  void setParamsUnchecked(Params const& params) { this->params_ = params; }

  Params params_;
};
}  // namespace lie

}  // namespace sophus
