// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Calculation of biinvariant means.

#pragma once

#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/rotation2.h"
#include "sophus/lie/rotation3.h"
#include "sophus/lie/scaling.h"
#include "sophus/lie/scaling_translation.h"
#include "sophus/lie/similarity2.h"
#include "sophus/lie/similarity3.h"
#include "sophus/lie/spiral_similarity2.h"
#include "sophus/lie/spiral_similarity3.h"
#include "sophus/lie/translation.h"

#include <complex>

namespace sophus {

/// Calculates mean iteratively.
///
/// Returns ``nullopt`` if it does not converge.
///
template <concepts::Range TSequenceContainer>
auto iterativeMean(
    TSequenceContainer const& foo_from_bar_transforms, int max_num_iterations)
    -> std::optional<typename TSequenceContainer::value_type> {
  size_t const k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));
  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");

  using Group = typename TSequenceContainer::value_type;
  using Scalar = typename Group::Scalar;
  using Tangent = Eigen::Vector<Scalar, Group::kDof>;

  // This implements the algorithm in the beginning of Sec. 4.2 in
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  Group foo_from_average = *std::begin(foo_from_bar_transforms);
  Scalar w = Scalar(1. / k_matrix_dim);
  for (int i = 0; i < max_num_iterations; ++i) {
    Tangent average;
    average.setZero();
    for (Group const& foo_from_bar : foo_from_bar_transforms) {
      average += w * (foo_from_average.inverse() * foo_from_bar).log();
    }
    Group foo_from_newaverage = foo_from_average * Group::exp(average);
    if (((foo_from_newaverage.inverse() * foo_from_average).log())
            .squaredNorm() < kEpsilon<Scalar>) {
      return foo_from_newaverage;
    }

    foo_from_average = foo_from_newaverage;
  }
  // LCOV_EXCL_START
  return std::nullopt;
  // LCOV_EXCL_STOP
}

#ifdef DOXYGEN_SHOULD_SKIP_THIS
/// Mean implementation for any Lie group.
template <class SequenceContainer, class Scalar>
optional<typename SequenceContainer::value_type> average(
    SequenceContainer const& foo_Ts_bar);
#else

// Mean implementation for Translation.
template <
    concepts::Range TSequenceContainer,
    int kPointDim = TSequenceContainer::value_type::kDof,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(TSequenceContainer const& foo_from_bar_transforms)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Translation<TScalar, kPointDim> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  size_t const k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));
  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");

  Eigen::Vector<TScalar, kPointDim> average;
  average.setZero();
  for (Translation<TScalar, kPointDim> const& foo_from_bar :
       foo_from_bar_transforms) {
    average += foo_from_bar.params();
  }
  return Translation<TScalar, kPointDim>::fromParams(
      average / TScalar(k_matrix_dim));
}

// Mean implementation for Scaling.
template <
    concepts::Range TSequenceContainer,
    int kPointDim = TSequenceContainer::value_type::kDof,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(TSequenceContainer const& foo_from_bar_transforms)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Scaling<TScalar, kPointDim> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  size_t const k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));
  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");

  Eigen::Array<TScalar, kPointDim, 1> average;
  average.setZero();
  for (Scaling<TScalar, kPointDim> const& foo_from_bar :
       foo_from_bar_transforms) {
    average += foo_from_bar.params().array().log();
  }
  return Scaling<TScalar, kPointDim>::fromParams(
      (average / TScalar(k_matrix_dim)).exp().matrix());
}

// Mean implementation for SO(2).
template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(TSequenceContainer const& foo_from_bar_transforms)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Rotation2<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  // This implements rotational part of Proposition 12 from Sec. 6.2 of
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  size_t const k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));
  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  Rotation2<TScalar> foo_from_average = *std::begin(foo_from_bar_transforms);
  TScalar w = TScalar(1. / k_matrix_dim);

  Eigen::Vector<TScalar, 1> average;
  average.setZero();
  for (Rotation2<TScalar> const& foo_from_bar : foo_from_bar_transforms) {
    average += w * (foo_from_average.inverse() * foo_from_bar).log();
  }
  return foo_from_average * Rotation2<TScalar>::exp(average);
}

// Mean implementation for RxSO(2).
template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(TSequenceContainer const& foo_from_bar_transforms)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            SpiralSimilarity2<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  size_t const k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));
  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  SpiralSimilarity2<TScalar> foo_from_average =
      *std::begin(foo_from_bar_transforms);
  TScalar w = TScalar(1. / k_matrix_dim);

  Eigen::Vector2<TScalar> average(TScalar(0), TScalar(0));
  for (SpiralSimilarity2<TScalar> const& foo_from_bar :
       foo_from_bar_transforms) {
    average += w * (foo_from_average.inverse() * foo_from_bar).log();
  }
  return foo_from_average * SpiralSimilarity2<TScalar>::exp(average);
}

namespace details {
template <class TScalar>
void getQuaternion(TScalar const&);

template <class TScalar>
auto getUnitQuaternion(Rotation3<TScalar> const& r)
    -> Eigen::Vector<TScalar, 4> {
  return r.params().template head<4>();
}

template <class TScalar>
auto getUnitQuaternion(SpiralSimilarity3<TScalar> const& s_r)
    -> Eigen::Vector<TScalar, 4> {
  return s_r.params().template head<4>().normalized();
}

template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto averageUnitQuaternion(TSequenceContainer const& foo_from_bar_transforms)
    -> Eigen::Vector<TScalar, 4> {
  // This:  http://stackoverflow.com/a/27410865/1221742
  size_t const k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));
  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  Eigen::Matrix<TScalar, 4, Eigen::Dynamic> q(4, k_matrix_dim);
  int i = 0;
  TScalar w = TScalar(1. / k_matrix_dim);
  for (auto const& foo_from_bar : foo_from_bar_transforms) {
    q.col(i) = w * details::getUnitQuaternion(foo_from_bar);
    ++i;
  }

  Eigen::Matrix<TScalar, 4, 4> q_qt = q * q.transpose();
  // TODO: Figure out why we can't use SelfAdjointEigenSolver here.
  Eigen::EigenSolver<Eigen::Matrix<TScalar, 4, 4> > es(q_qt);

  std::complex<TScalar> max_eigenvalue = es.eigenvalues()[0];
  Eigen::Matrix<std::complex<TScalar>, 4, 1> max_eigenvector =
      es.eigenvectors().col(0);

  for (int i = 1; i < 4; i++) {
    if (std::norm(es.eigenvalues()[i]) > std::norm(max_eigenvalue)) {
      max_eigenvalue = es.eigenvalues()[i];
      max_eigenvector = es.eigenvectors().col(i);
    }
  }
  Eigen::Vector<TScalar, 4> quat;
  quat <<                         //
      max_eigenvector[0].real(),  //
      max_eigenvector[1].real(),  //
      max_eigenvector[2].real(),  //
      max_eigenvector[3].real();
  return quat;
}
}  // namespace details

// Mean implementation for SO(3).
//
// TODO: Detect degenerated cases and return nullopt.
template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(TSequenceContainer const& foo_from_bar_transforms)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Rotation3<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  return Rotation3<TScalar>::fromParams(
      details::averageUnitQuaternion(foo_from_bar_transforms));
}

// Mean implementation for R x SO(3).
template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(TSequenceContainer const& foo_from_bar_transforms)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            SpiralSimilarity3<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  size_t k_matrix_dim = std::distance(
      std::begin(foo_from_bar_transforms), std::end(foo_from_bar_transforms));

  SOPHUS_ASSERT(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  TScalar scale_sum = TScalar(0);
  using std::exp;
  using std::log;
  for (SpiralSimilarity3<TScalar> const& foo_from_bar :
       foo_from_bar_transforms) {
    scale_sum += log(foo_from_bar.scale());
  }
  return SpiralSimilarity3<TScalar>(
      Rotation3<TScalar>::fromParams(
          details::averageUnitQuaternion(foo_from_bar_transforms)),
      exp(scale_sum / TScalar(k_matrix_dim)));
}

template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(
    TSequenceContainer const& foo_from_bar_transforms,
    int max_num_iterations = 20)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Isometry2<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  // TODO: Implement Proposition 12 from Sec. 6.2 of
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  return iterativeMean(foo_from_bar_transforms, max_num_iterations);
}

template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(
    TSequenceContainer const& foo_from_bar_transforms,
    int max_num_iterations = 20)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Similarity2<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  return iterativeMean(foo_from_bar_transforms, max_num_iterations);
}

template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(
    TSequenceContainer const& foo_from_bar_transforms,
    int max_num_iterations = 20)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Isometry3<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  return iterativeMean(foo_from_bar_transforms, max_num_iterations);
}

template <
    concepts::Range TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(
    TSequenceContainer const& foo_from_bar_transforms,
    int max_num_iterations = 20)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            Similarity3<TScalar> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  return iterativeMean(foo_from_bar_transforms, max_num_iterations);
}

template <
    concepts::Range TSequenceContainer,
    int kPointDim = TSequenceContainer::value_type::kPointDim,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
auto average(
    TSequenceContainer const& foo_from_bar_transforms,
    int max_num_iterations = 20)
    -> std::enable_if_t<
        std::is_same<
            typename TSequenceContainer::value_type,
            ScalingTranslation<TScalar, kPointDim> >::value,
        std::optional<typename TSequenceContainer::value_type> > {
  return iterativeMean(foo_from_bar_transforms, max_num_iterations);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace sophus
