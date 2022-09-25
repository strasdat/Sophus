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

#include "sophus/common/common.h"
#include "sophus/lie/cartesian.h"
#include "sophus/lie/rxso2.h"
#include "sophus/lie/rxso3.h"
#include "sophus/lie/se2.h"
#include "sophus/lie/se3.h"
#include "sophus/lie/sim2.h"
#include "sophus/lie/sim3.h"
#include "sophus/lie/so2.h"
#include "sophus/lie/so3.h"

#include <complex>

namespace sophus {

/// Calculates mean iteratively.
///
/// Returns ``nullopt`` if it does not converge.
///
template <class TSequenceContainer>
std::optional<typename TSequenceContainer::value_type> iterativeMean(
    TSequenceContainer const& foo_transforms_bar, int max_num_iterations) {
  size_t k_matrix_dim = foo_transforms_bar.size();
  FARM_CHECK(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");

  using Group = typename TSequenceContainer::value_type;
  using Scalar = typename Group::Scalar;
  using Tangent = typename Group::Tangent;

  // This implements the algorithm in the beginning of Sec. 4.2 in
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  Group foo_transform_average = foo_transforms_bar.front();
  Scalar w = Scalar(1. / k_matrix_dim);
  for (int i = 0; i < max_num_iterations; ++i) {
    Tangent average;
    setToZero<Tangent>(average);
    for (Group const& foo_transform_bar : foo_transforms_bar) {
      average +=
          w * (foo_transform_average.inverse() * foo_transform_bar).log();
    }
    Group foo_transform_newaverage =
        foo_transform_average * Group::exp(average);
    if (squaredNorm<Tangent>(
            (foo_transform_newaverage.inverse() * foo_transform_average)
                .log()) < kEpsilon<Scalar>) {
      return foo_transform_newaverage;
    }

    foo_transform_average = foo_transform_newaverage;
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

// Mean implementation for Cartesian.
template <
    class TSequenceContainer,
    int kPointDim = TSequenceContainer::value_type::kDoF,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<
        typename TSequenceContainer::value_type,
        Cartesian<TScalar, kPointDim> >::value,
    std::optional<typename TSequenceContainer::value_type> >
average(TSequenceContainer const& foo_transforms_bar) {
  size_t k_matrix_dim = std::distance(
      std::begin(foo_transforms_bar), std::end(foo_transforms_bar));
  FARM_CHECK(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");

  Eigen::Vector<TScalar, kPointDim> average;
  average.setZero();
  for (Cartesian<TScalar, kPointDim> const& foo_transform_bar :
       foo_transforms_bar) {
    average += foo_transform_bar.params();
  }
  return Cartesian<TScalar, kPointDim>(average / TScalar(k_matrix_dim));
}

// Mean implementation for SO(2).
template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, So2<TScalar> >::value,
    std::optional<typename TSequenceContainer::value_type> >
average(TSequenceContainer const& foo_transforms_bar) {
  // This implements rotational part of Proposition 12 from Sec. 6.2 of
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  size_t k_matrix_dim = std::distance(
      std::begin(foo_transforms_bar), std::end(foo_transforms_bar));
  FARM_CHECK(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  So2<TScalar> foo_transform_average = foo_transforms_bar.front();
  TScalar w = TScalar(1. / k_matrix_dim);

  TScalar average(0);
  for (So2<TScalar> const& foo_transform_bar : foo_transforms_bar) {
    average += w * (foo_transform_average.inverse() * foo_transform_bar).log();
  }
  return foo_transform_average * So2<TScalar>::exp(average);
}

// Mean implementation for RxSO(2).
template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, RxSo2<TScalar> >::
        value,
    std::optional<typename TSequenceContainer::value_type> >
average(TSequenceContainer const& foo_transforms_bar) {
  size_t k_matrix_dim = std::distance(
      std::begin(foo_transforms_bar), std::end(foo_transforms_bar));
  FARM_CHECK(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  RxSo2<TScalar> foo_transform_average = foo_transforms_bar.front();
  TScalar w = TScalar(1. / k_matrix_dim);

  Eigen::Vector2<TScalar> average(TScalar(0), TScalar(0));
  for (RxSo2<TScalar> const& foo_transform_bar : foo_transforms_bar) {
    average += w * (foo_transform_average.inverse() * foo_transform_bar).log();
  }
  return foo_transform_average * RxSo2<TScalar>::exp(average);
}

namespace details {
template <class TScalar>
void getQuaternion(TScalar const&);

template <class TScalar>
Eigen::Quaternion<TScalar> getUnitQuaternion(So3<TScalar> const& r) {
  return r.unitQuaternion();
}

template <class TScalar>
Eigen::Quaternion<TScalar> getUnitQuaternion(RxSo3<TScalar> const& s_r) {
  return s_r.so3().unitQuaternion();
}

template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
Eigen::Quaternion<TScalar> averageUnitQuaternion(
    TSequenceContainer const& foo_transforms_bar) {
  // This:  http://stackoverflow.com/a/27410865/1221742
  size_t k_matrix_dim = std::distance(
      std::begin(foo_transforms_bar), std::end(foo_transforms_bar));
  FARM_CHECK(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  Eigen::Matrix<TScalar, 4, Eigen::Dynamic> q(4, k_matrix_dim);
  int i = 0;
  TScalar w = TScalar(1. / k_matrix_dim);
  for (auto const& foo_transform_bar : foo_transforms_bar) {
    q.col(i) = w * details::getUnitQuaternion(foo_transform_bar).coeffs();
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
  Eigen::Quaternion<TScalar> quat;
  quat.coeffs() <<                //
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
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, So3<TScalar> >::value,
    std::optional<typename TSequenceContainer::value_type> >
average(TSequenceContainer const& foo_transforms_bar) {
  return So3<TScalar>(details::averageUnitQuaternion(foo_transforms_bar));
}

// Mean implementation for R x SO(3).
template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, RxSo3<TScalar> >::
        value,
    std::optional<typename TSequenceContainer::value_type> >
average(TSequenceContainer const& foo_transforms_bar) {
  size_t k_matrix_dim = std::distance(
      std::begin(foo_transforms_bar), std::end(foo_transforms_bar));

  FARM_CHECK(k_matrix_dim >= 1, "kMatrixDim must be >= 1.");
  TScalar scale_sum = TScalar(0);
  using std::exp;
  using std::log;
  for (RxSo3<TScalar> const& foo_transform_bar : foo_transforms_bar) {
    scale_sum += log(foo_transform_bar.scale());
  }
  return RxSo3<TScalar>(
      exp(scale_sum / TScalar(k_matrix_dim)),
      So3<TScalar>(details::averageUnitQuaternion(foo_transforms_bar)));
}

template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, Se2<TScalar> >::value,
    std::optional<typename TSequenceContainer::value_type> >
average(
    TSequenceContainer const& foo_transforms_bar, int max_num_iterations = 20) {
  // TODO: Implement Proposition 12 from Sec. 6.2 of
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  return iterativeMean(foo_transforms_bar, max_num_iterations);
}

template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, Sim2<TScalar> >::
        value,
    std::optional<typename TSequenceContainer::value_type> >
average(
    TSequenceContainer const& foo_transforms_bar, int max_num_iterations = 20) {
  return iterativeMean(foo_transforms_bar, max_num_iterations);
}

template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, Se3<TScalar> >::value,
    std::optional<typename TSequenceContainer::value_type> >
average(
    TSequenceContainer const& foo_transforms_bar, int max_num_iterations = 20) {
  return iterativeMean(foo_transforms_bar, max_num_iterations);
}

template <
    class TSequenceContainer,
    class TScalar = typename TSequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename TSequenceContainer::value_type, Sim3<TScalar> >::
        value,
    std::optional<typename TSequenceContainer::value_type> >
average(
    TSequenceContainer const& foo_transforms_bar, int max_num_iterations = 20) {
  return iterativeMean(foo_transforms_bar, max_num_iterations);
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace sophus
