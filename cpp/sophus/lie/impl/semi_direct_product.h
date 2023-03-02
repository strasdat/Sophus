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
#include "sophus/linalg/unit_vector.h"
#include "sophus/linalg/vector_space.h"

namespace sophus {
namespace lie {

template <
    class TScalar,
    int kTranslationDim,
    template <class, int>
    class TSubGroup>
requires concepts::LieSubgroupImpl<TSubGroup<TScalar, kTranslationDim>>
class SemiDirectProductWithTranslation {
 public:
  using Scalar = TScalar;
  using SubGroup = TSubGroup<Scalar, kTranslationDim>;

  // The is also the dimension of the translation.
  static int const kPointDim = kTranslationDim;
  static_assert(kPointDim == SubGroup::kPointDim);
  static_assert(kPointDim == SubGroup::kAmbientDim);

  // Non-zero translation shift the coordinate origin.
  static bool constexpr kIsOriginPreserving = false;
  static_assert(
      static_cast<bool>(SubGroup::kIsOriginPreserving),
      "The subgroup is origin preserving by definition.");
  static bool constexpr kIsAxisDirectionPreserving =
      SubGroup::kIsAxisDirectionPreserving;
  static bool constexpr kIsDirectionVectorPreserving =
      SubGroup::kIsDirectionVectorPreserving;
  static bool constexpr kIsShapePreserving = SubGroup::kIsShapePreserving;
  static bool constexpr kIsSizePreserving = SubGroup::kIsSizePreserving;

  static_assert(
      static_cast<bool>(SubGroup::kIisParallelLinePreserving),
      "The subgroup by definition needs to map parallel lines to parallel "
      "lines. In particular projective mappings are not allowed.");
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = SubGroup::kDof + kPointDim;
  static int const kNumParams = SubGroup::kNumParams + kPointDim;
  static int const kAmbientDim = kPointDim + 1;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return params(
        SubGroup::identityParams(), Eigen::Vector<Scalar, kPointDim>::Zero());
  }

  static auto areParamsValid(Eigen::Vector<Scalar, kNumParams> const& params)
      -> sophus::Expected<Success> {
    return SubGroup::areParamsValid(subgroupParams(params));
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> tangent)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        SubGroup::exp(subgroupTangent(tangent));
    return params(
        subgroup_params,
        (SubGroup::matV(subgroup_params, subgroupTangent(tangent)) *
         translationTangent(tangent))
            .eval());
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, SubGroup::kDof> subgroup_tangent =
        SubGroup::log(subgroupParams(params));
    return tangent(
        SubGroup::matVInverse(subgroupParams(params), subgroup_tangent) *
            translation(params),
        subgroup_tangent);
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> hat_mat;
    hat_mat.setZero();
    hat_mat.template topLeftCorner<kPointDim, kPointDim>() =
        SubGroup::hat(subgroupTangent(tangent));
    hat_mat.template topRightCorner<kPointDim, 1>() =
        translationTangent(tangent);
    return hat_mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return tangent(
        mat.template topRightCorner<kPointDim, 1>().eval(),
        SubGroup::vee(
            mat.template topLeftCorner<kPointDim, kPointDim>().eval()));
  }

  // group operations
  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return SemiDirectProductWithTranslation::params(
        SubGroup::multiplication(
            subgroupParams(lhs_params), subgroupParams(rhs_params)),
        SubGroup::action(subgroupParams(lhs_params), translation(rhs_params)) +
            translation(lhs_params));
  }

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        SubGroup::inverse(subgroupParams(params));
    return SemiDirectProductWithTranslation::params(
        subgroup_params,
        (-SubGroup::action(subgroup_params, translation(params))).eval());
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return SubGroup::action(subgroupParams(params), point) +
           translation(params);
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return unproj(point);
  }

  // Matrices

  static auto compactMatrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kPointDim, kAmbientDim> mat;
    mat.template topLeftCorner<kPointDim, kPointDim>() =
        SubGroup::compactMatrix(subgroupParams(params));
    mat.template topRightCorner<kPointDim, 1>() = translation(params);
    return mat;
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    mat.template topLeftCorner<kPointDim, kAmbientDim>() =
        compactMatrix(params);
    mat(kPointDim, kPointDim) = 1.0;
    return mat;
  }

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      UnitVector<Scalar, kPointDim> const& direction_vector)
      -> UnitVector<Scalar, kPointDim> {
    return SubGroup::action(subgroupParams(params), direction_vector);
  }

  // derivatives
  static auto adj(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adjoint;

    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        subgroupParams(params);

    mat_adjoint.template topLeftCorner<kPointDim, kPointDim>() =
        SubGroup::matrix(subgroup_params);
    mat_adjoint.template topRightCorner<kPointDim, SubGroup::kDof>() =
        SubGroup::topRightAdj(subgroup_params, translation(params));

    mat_adjoint.template bottomLeftCorner<SubGroup::kDof, kPointDim>()
        .setZero();
    mat_adjoint.template bottomRightCorner<SubGroup::kDof, SubGroup::kDof>() =
        SubGroup::adj(subgroup_params);

    return mat_adjoint;
  }

  // static auto dxExpX(Eigen::Vector<Scalar, kDof> const& tangent)
  //     -> Eigen::Matrix<Scalar, kNumParams, kDof> {
  //   Eigen::Matrix<Scalar, kNumParams, kDof> j;
  //   j.setZero();
  //   j.template topRightCorner<SubGroup::kNumParams, SubGroup::kDof>() =
  //       SubGroup::dxExpX(subgroupTangent(tangent));
  //   j.template bottomLeftCorner<kPointDim, kPointDim>().setIdentity();
  //   return j;
  // }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, kNumParams, kDof> j;
    j.setZero();
    j.template topRightCorner<SubGroup::kNumParams, SubGroup::kDof>() =
        SubGroup::dxExpXAt0();
    j.template bottomLeftCorner<kPointDim, kPointDim>().setIdentity();
    return j;
  }

  static auto dxExpXTimesPointAt0(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, kPointDim, kDof> j;
    j.template topLeftCorner<kPointDim, kPointDim>().setIdentity();
    j.template topRightCorner<kPointDim, SubGroup::kDof>() =
        SubGroup::dxExpXTimesPointAt0(point);
    return j;
  }

  static auto dxThisMulExpXAt0(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, kNumParams, kDof> j;
    j.setZero();
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        subgroupParams(params);
    j.template topRightCorner<SubGroup::kNumParams, SubGroup::kDof>() =
        SubGroup::dxThisMulExpXAt0(subgroup_params);
    j.template bottomLeftCorner<kPointDim, kPointDim>() =
        SubGroup::matrix(subgroup_params);
    return j;
  }

  static auto dxLogThisInvTimesXAtThis(
      Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    Eigen::Matrix<Scalar, kDof, kNumParams> j;
    j.setZero();
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        subgroupParams(params);
    j.template bottomLeftCorner<SubGroup::kDof, SubGroup::kNumParams>() =
        SubGroup::dxLogThisInvTimesXAtThis(subgroup_params);
    j.template topRightCorner<kPointDim, kPointDim>() =
        SubGroup::matrix(subgroup_params).inverse();
    return j;
  }

  // for tests

  static auto tangentExamples() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    std::vector<Eigen::Vector<Scalar, kDof>> examples;
    for (auto const& group_tangent : SubGroup::tangentExamples()) {
      for (auto const& translation_tangents : exampleTranslations()) {
        examples.push_back(tangent(translation_tangents, group_tangent));
      }
    }
    return examples;
  }

  static auto paramsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    std::vector<Eigen::Vector<Scalar, kNumParams>> examples;
    for (auto const& subgroup_params : SubGroup::paramsExamples()) {
      for (auto const& right_params : exampleTranslations()) {
        examples.push_back(params(subgroup_params, right_params));
      }
    }
    return examples;
  }

  static auto invalidParamsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    std::vector<Eigen::Vector<Scalar, kNumParams>> examples;
    for (auto const& subgroup_params : SubGroup::invalidParamsExamples()) {
      for (auto const& right_params : exampleTranslations()) {
        examples.push_back(params(subgroup_params, right_params));
      }
    }
    return examples;
  }

 private:
  static auto subgroupParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, SubGroup::kNumParams> {
    return params.template head<SubGroup::kNumParams>();
  }

  static auto translation(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kPointDim> {
    return params.template tail<kPointDim>();
  }

  static auto params(
      Eigen::Vector<Scalar, SubGroup::kNumParams> const& sub_group_params,
      Eigen::Vector<Scalar, kPointDim> const& translation)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, kNumParams> params;
    params.template head<SubGroup::kNumParams>() = sub_group_params;
    params.template tail<kPointDim>() = translation;
    return params;
  }

  static auto subgroupTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, SubGroup::kDof> {
    return tangent.template tail<SubGroup::kDof>();
  }

  static auto translationTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, kPointDim> {
    return tangent.template head<kPointDim>();
  }

  static auto tangent(
      Eigen::Vector<Scalar, kPointDim> const& translation,
      Eigen::Vector<Scalar, SubGroup::kDof> const& subgroup_tangent)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, kDof> tangent;
    tangent.template head<kPointDim>() = translation;
    tangent.template tail<SubGroup::kDof>() = subgroup_tangent;
    return tangent;
  }

  static auto exampleTranslations()
      -> std::vector<Eigen::Vector<Scalar, kPointDim>> {
    return ::sophus::pointExamples<Scalar, kPointDim>();
  }
};

}  // namespace lie
}  // namespace sophus
