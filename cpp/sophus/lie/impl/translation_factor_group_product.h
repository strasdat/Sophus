// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/lie_group.h"
#include "sophus/linalg/homogeneous.h"
#include "sophus/linalg/vector_space.h"
#include "sophus/manifold/unit_vector.h"

namespace sophus {
namespace lie {

/// Semi direct product of a Lie group (factor group) and the
/// vector space (translation).
template <
    class TScalar,
    int kTranslationDim,
    template <class>
    class TFactorGroup>
requires concepts::LieFactorGroupImpl<TFactorGroup<TScalar>>
class TranslationFactorGroupProduct {
 public:
  using Scalar = TScalar;
  using FactorGroup = TFactorGroup<Scalar>;

  // The is also the dimension of the translation.
  static int const kPointDim = kTranslationDim;
  static_assert(kPointDim == FactorGroup::kPointDim);
  static_assert(kPointDim == FactorGroup::kAmbientDim);

  // Non-zero translation shift the coordinate origin.
  static bool constexpr kIsOriginPreserving = false;
  static_assert(
      static_cast<bool>(FactorGroup::kIsOriginPreserving),
      "The factor group is origin preserving by definition.");
  static bool constexpr kIsAxisDirectionPreserving =
      FactorGroup::kIsAxisDirectionPreserving;
  static bool constexpr kIsDirectionVectorPreserving =
      FactorGroup::kIsDirectionVectorPreserving;
  static bool constexpr kIsShapePreserving = FactorGroup::kIsShapePreserving;
  static bool constexpr kIsSizePreserving = FactorGroup::kIsSizePreserving;

  static_assert(
      static_cast<bool>(FactorGroup::kIisParallelLinePreserving),
      "The factor group by definition needs to map parallel lines to parallel "
      "lines. In particular projective mappings are not allowed.");
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = FactorGroup::kDof + kPointDim;
  static int const kNumParams = FactorGroup::kNumParams + kPointDim;
  static int const kAmbientDim = kPointDim + 1;

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;

  template <class TCompatibleScalar>
  using ScalarReturn = typename Eigen::
      ScalarBinaryOpTraits<Scalar, TCompatibleScalar>::ReturnType;

  template <class TCompatibleScalar>
  using ParamsReturn =
      Eigen::Vector<ScalarReturn<TCompatibleScalar>, kNumParams>;

  template <class TCompatibleScalar>
  using PointReturn = Eigen::Vector<ScalarReturn<TCompatibleScalar>, kPointDim>;

  template <class TCompatibleScalar>
  using UnitVectorReturn =
      UnitVector<ScalarReturn<TCompatibleScalar>, kPointDim>;

  // constructors and factories

  static auto identityParams() -> Params {
    return params(FactorGroup::identityParams(), Point::Zero().eval());
  }

  static auto areParamsValid(Params const& params)
      -> sophus::Expected<Success> {
    return FactorGroup::areParamsValid(factorGroupParams(params));
  }

  static auto hasShortestPathAmbiguity(Params const& params) -> bool {
    return FactorGroup::hasShortestPathAmbiguity(factorGroupParams(params));
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent tangent) -> Params {
    Eigen::Vector<Scalar, FactorGroup::kNumParams> factor_group_params =
        FactorGroup::exp(factorTangent(tangent));
    return params(
        factor_group_params,
        (FactorGroup::matV(factor_group_params, factorTangent(tangent)) *
         translationTangent(tangent))
            .eval());
  }

  static auto log(Params const& params) -> Tangent {
    Eigen::Vector<Scalar, FactorGroup::kDof> factor_tangent =
        FactorGroup::log(factorGroupParams(params));
    return tangent(
        FactorGroup::matVInverse(factorGroupParams(params), factor_tangent) *
            translation(params),
        factor_tangent);
  }

  static auto hat(Tangent const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> hat_mat;
    hat_mat.setZero();
    hat_mat.template topLeftCorner<kPointDim, kPointDim>() =
        FactorGroup::hat(factorTangent(tangent));
    hat_mat.template topRightCorner<kPointDim, 1>() =
        translationTangent(tangent);
    return hat_mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return tangent(
        mat.template topRightCorner<kPointDim, 1>().eval(),
        FactorGroup::vee(
            mat.template topLeftCorner<kPointDim, kPointDim>().eval()));
  }

  // group operations
  template <class TCompatibleScalar>
  static auto multiplication(
      Params const& lhs_params,
      Eigen::Vector<TCompatibleScalar, kNumParams> const& rhs_params)
      -> ParamsReturn<TCompatibleScalar> {
    typename FactorGroup::template ParamsReturn<TCompatibleScalar> p =
        FactorGroup::multiplication(
            factorGroupParams(lhs_params), factorGroupParams(rhs_params));

    typename FactorGroup::template PointReturn<TCompatibleScalar> t =
        FactorGroup::action(
            factorGroupParams(lhs_params), translation(rhs_params)) +
        translation(lhs_params);

    return TranslationFactorGroupProduct::params(p, t);
  }

  static auto inverse(Params const& params) -> Params {
    Eigen::Vector<Scalar, FactorGroup::kNumParams> factor_group_params =
        FactorGroup::inverse(factorGroupParams(params));
    return TranslationFactorGroupProduct::params(
        factor_group_params,
        (-FactorGroup::action(factor_group_params, translation(params)))
            .eval());
  }

  // Point actions

  template <class TCompatibleScalar>
  static auto action(
      Params const& params,
      Eigen::Matrix<TCompatibleScalar, kPointDim, 1> const& point)
      -> PointReturn<TCompatibleScalar> {
    return FactorGroup::action(factorGroupParams(params), point) +
           translation(params);
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return unproj(point);
  }

  template <class TCompatibleScalar>
  static auto action(
      Params const& params,
      UnitVector<TCompatibleScalar, kPointDim> const& direction)
      -> UnitVectorReturn<TCompatibleScalar> {
    return FactorGroup::action(factorGroupParams(params), direction);
  }

  static auto adj(Params const& params) -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adjoint;

    Eigen::Vector<Scalar, FactorGroup::kNumParams> factor_group_params =
        factorGroupParams(params);

    mat_adjoint.template topLeftCorner<kPointDim, kPointDim>() =
        FactorGroup::matrix(factor_group_params);
    mat_adjoint.template topRightCorner<kPointDim, FactorGroup::kDof>() =
        FactorGroup::adjOfTranslation(factor_group_params, translation(params));

    mat_adjoint.template bottomLeftCorner<FactorGroup::kDof, kPointDim>()
        .setZero();
    mat_adjoint
        .template bottomRightCorner<FactorGroup::kDof, FactorGroup::kDof>() =
        FactorGroup::adj(factor_group_params);

    return mat_adjoint;
  }

  static auto ad(Tangent const& tangent) -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> ad;
    ad.template topLeftCorner<kPointDim, kPointDim>() =
        FactorGroup::hat(factorTangent(tangent));
    ad.template topRightCorner<kPointDim, FactorGroup::kDof>() =
        FactorGroup::adOfTranslation(translationTangent(tangent));

    ad.template bottomLeftCorner<FactorGroup::kDof, kPointDim>().setZero();

    ad.template bottomRightCorner<FactorGroup::kDof, FactorGroup::kDof>() =
        FactorGroup::ad(factorTangent(tangent));

    return ad;
  }

  // Matrices

  static auto compactMatrix(Params const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kPointDim, kAmbientDim> mat;
    mat.template topLeftCorner<kPointDim, kPointDim>() =
        FactorGroup::compactMatrix(factorGroupParams(params));
    mat.template topRightCorner<kPointDim, 1>() = translation(params);
    return mat;
  }

  static auto matrix(Params const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    mat.template topLeftCorner<kPointDim, kAmbientDim>() =
        compactMatrix(params);
    mat(kPointDim, kPointDim) = 1.0;
    return mat;
  }

  // derivatives

  // static auto dxExpX(Tangent const& tangent)
  //     -> Eigen::Matrix<Scalar, kNumParams, kDof> {
  //   Eigen::Matrix<Scalar, kNumParams, kDof> j;
  //   j.setZero();
  //   j.template topRightCorner<FactorGroup::kNumParams, FactorGroup::kDof>() =
  //       FactorGroup::dxExpX(factorTangent(tangent));
  //   j.template bottomLeftCorner<kPointDim, kPointDim>().setIdentity();
  //   return j;
  // }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, kNumParams, kDof> j;
    j.setZero();
    j.template topRightCorner<FactorGroup::kNumParams, FactorGroup::kDof>() =
        FactorGroup::dxExpXAt0();
    j.template bottomLeftCorner<kPointDim, kPointDim>().setIdentity();
    return j;
  }

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, kPointDim, kDof> j;
    j.template topLeftCorner<kPointDim, kPointDim>().setIdentity();
    j.template topRightCorner<kPointDim, FactorGroup::kDof>() =
        FactorGroup::dxExpXTimesPointAt0(point);
    return j;
  }

  static auto dxThisMulExpXAt0(Params const& params)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, kNumParams, kDof> j;
    j.setZero();
    Eigen::Vector<Scalar, FactorGroup::kNumParams> factor_group_params =
        factorGroupParams(params);
    j.template topRightCorner<FactorGroup::kNumParams, FactorGroup::kDof>() =
        FactorGroup::dxThisMulExpXAt0(factor_group_params);
    j.template bottomLeftCorner<kPointDim, kPointDim>() =
        FactorGroup::matrix(factor_group_params);
    return j;
  }

  static auto dxLogThisInvTimesXAtThis(Params const& params)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    Eigen::Matrix<Scalar, kDof, kNumParams> j;
    j.setZero();
    Eigen::Vector<Scalar, FactorGroup::kNumParams> factor_group_params =
        factorGroupParams(params);
    j.template bottomLeftCorner<FactorGroup::kDof, FactorGroup::kNumParams>() =
        FactorGroup::dxLogThisInvTimesXAtThis(factor_group_params);
    j.template topRightCorner<kPointDim, kPointDim>() =
        FactorGroup::matrix(factor_group_params).inverse();
    return j;
  }

  // for tests

  static auto tangentExamples() -> std::vector<Tangent> {
    std::vector<Tangent> examples;
    for (auto const& group_tangent : FactorGroup::tangentExamples()) {
      for (auto const& translation_tangents : exampleTranslations()) {
        examples.push_back(tangent(translation_tangents, group_tangent));
      }
    }
    return examples;
  }

  static auto paramsExamples() -> std::vector<Params> {
    std::vector<Params> examples;
    for (auto const& factor_group_params : FactorGroup::paramsExamples()) {
      for (auto const& right_params : exampleTranslations()) {
        examples.push_back(params(factor_group_params, right_params));
      }
    }
    return examples;
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    std::vector<Params> examples;
    for (auto const& factor_group_params :
         FactorGroup::invalidParamsExamples()) {
      for (auto const& right_params : exampleTranslations()) {
        examples.push_back(params(factor_group_params, right_params));
      }
    }
    return examples;
  }

 private:
  template <class TScalar2>
  static auto factorGroupParams(
      Eigen::Vector<TScalar2, kNumParams> const& params)
      -> Eigen::Vector<TScalar2, FactorGroup::kNumParams> {
    return params.template head<FactorGroup::kNumParams>();
  }

  template <class TScalar2>
  static auto translation(Eigen::Vector<TScalar2, kNumParams> const& params)
      -> Eigen::Vector<TScalar2, kPointDim> {
    return params.template tail<kPointDim>();
  }

  template <class TScalar2>
  static auto params(
      Eigen::Vector<TScalar2, FactorGroup::kNumParams> const&
          factor_grop_tarams,
      Eigen::Vector<TScalar2, kPointDim> const& translation)
      -> Eigen::Vector<TScalar2, kNumParams> {
    Eigen::Vector<TScalar2, kNumParams> params;
    params.template head<FactorGroup::kNumParams>() = factor_grop_tarams;
    params.template tail<kPointDim>() = translation;
    return params;
  }

  static auto factorTangent(Tangent const& tangent)
      -> Eigen::Vector<Scalar, FactorGroup::kDof> {
    return tangent.template tail<FactorGroup::kDof>();
  }

  static auto translationTangent(Tangent const& tangent) -> Point {
    return tangent.template head<kPointDim>();
  }

  static auto tangent(
      Point const& translation,
      Eigen::Vector<Scalar, FactorGroup::kDof> const& factor_tangent)
      -> Tangent {
    Tangent tangent;
    tangent.template head<kPointDim>() = translation;
    tangent.template tail<FactorGroup::kDof>() = factor_tangent;
    return tangent;
  }

  static auto exampleTranslations() -> std::vector<Point> {
    return ::sophus::pointExamples<Scalar, kPointDim>();
  }
};

template <int kTranslationDim, template <class> class TFactorGroup>
struct WithDimAndSubgroup {
  template <class TScalar>
  using SemiDirectProduct =
      TranslationFactorGroupProduct<TScalar, kTranslationDim, TFactorGroup>;
};

}  // namespace lie
}  // namespace sophus
