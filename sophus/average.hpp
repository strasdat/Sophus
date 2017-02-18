#ifndef SOPHUS_MEAN_H
#define SOPHUS_MEAN_H

#include "common.hpp"
#include "rxso3.hpp"
#include "se2.hpp"
#include "se3.hpp"
#include "sim3.hpp"
#include "so2.hpp"
#include "so3.hpp"

namespace Sophus {

template <class SequenceContainer>
optional<typename SequenceContainer::value_type> iterativeMean(
    SequenceContainer const& foo_Ts_bar, int max_num_iterations) {
  size_t N = foo_Ts_bar.size();
  SOPHUS_ENSURE(N >= 1, "N must be >= 1.");

  using Group = typename SequenceContainer::value_type;
  using Scalar = typename Group::Scalar;
  using Tangent = typename Group::Tangent;

  // This implements the algorithm in the beginning of Sec. 4.2 in
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  Group foo_T_average = foo_Ts_bar.front();
  Scalar w = Scalar(1. / N);
  for (int i = 0; i < max_num_iterations; ++i) {
    Tangent average;
    setToZero<Tangent>(average);
    for (Group const& foo_T_bar : foo_Ts_bar) {
      average += w * (foo_T_average.inverse() * foo_T_bar).log();
    }
    Group foo_T_newaverage = foo_T_average * Group::exp(average);
    if (squaredNorm<Tangent>(
            (foo_T_newaverage.inverse() * foo_T_average).log()) <
        Constants<Scalar>::epsilon()) {
      return foo_T_newaverage;
    }

    foo_T_average = foo_T_newaverage;
  }
  return nullopt;
}

// Mean implementation for SO(2).
template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename SequenceContainer::value_type, SO2<Scalar>>::value,
    optional<typename SequenceContainer::value_type>>
average(SequenceContainer const& foo_Ts_bar) {
  // This implements rotational part of Proposition 12 from Sec. 6.2 of
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  size_t N = std::distance(std::begin(foo_Ts_bar), std::end(foo_Ts_bar));
  SOPHUS_ENSURE(N >= 1, "N must be >= 1.");
  SO2<Scalar> foo_T_average = foo_Ts_bar.front();
  Scalar w = Scalar(1. / N);

  Scalar average(0);
  for (SO2<Scalar> const& foo_T_bar : foo_Ts_bar) {
    average += w * (foo_T_average.inverse() * foo_T_bar).log();
  }
  return foo_T_average * SO2<Scalar>::exp(average);
}

namespace details {
template <class T>
void getQuaternion(T const&);

template <class Scalar>
Eigen::Quaternion<Scalar> getUnitQuaternion(SO3<Scalar> const& R) {
  return R.unit_quaternion();
}

template <class Scalar>
Eigen::Quaternion<Scalar> getUnitQuaternion(RxSO3<Scalar> const& sR) {
  return sR.so3().unit_quaternion();
}

template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
Eigen::Quaternion<Scalar> averageUnitQuaternion(
    SequenceContainer const& foo_Ts_bar) {
  // This:  http://stackoverflow.com/a/27410865/1221742
  size_t N = std::distance(std::begin(foo_Ts_bar), std::end(foo_Ts_bar));
  SOPHUS_ENSURE(N >= 1, "N must be >= 1.");
  Eigen::Matrix<Scalar, 4, Eigen::Dynamic> Q(4, N);
  int i = 0;
  Scalar w = Scalar(1. / N);
  for (auto const& foo_T_bar : foo_Ts_bar) {
    Q.col(i) = w * details::getUnitQuaternion(foo_T_bar).coeffs();
    ++i;
  }

  Eigen::Matrix<Scalar, 4, 4> QQt = Q * Q.transpose();
  // TODO: Figure out why we can't use SelfAdjointEigenSolver here.
  Eigen::EigenSolver<Eigen::Matrix<Scalar, 4, 4>> es(QQt);

  std::complex<Scalar> max_eigenvalue = es.eigenvalues()[0];
  Eigen::Matrix<std::complex<Scalar>, 4, 1> max_eigenvector =
      es.eigenvectors().col(0);

  for (int i = 1; i < 4; i++) {
    if (std::norm(es.eigenvalues()[i]) > std::norm(max_eigenvalue)) {
      max_eigenvalue = es.eigenvalues()[i];
      max_eigenvector = es.eigenvectors().col(i);
    }
  }
  Eigen::Quaternion<Scalar> quat;
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
template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename SequenceContainer::value_type, SO3<Scalar>>::value,
    optional<typename SequenceContainer::value_type>>
average(SequenceContainer const& foo_Ts_bar) {
  return SO3<Scalar>(details::averageUnitQuaternion(foo_Ts_bar));
}

// Mean implementation for R x SO(3).
template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename SequenceContainer::value_type, RxSO3<Scalar>>::value,
    optional<typename SequenceContainer::value_type>>
average(SequenceContainer const& foo_Ts_bar) {
  size_t N = std::distance(std::begin(foo_Ts_bar), std::end(foo_Ts_bar));

  SOPHUS_ENSURE(N >= 1, "N must be >= 1.");
  Scalar scale_sum = Scalar(0);
  using std::log;
  using std::exp;
  for (RxSO3<Scalar> const& foo_T_bar : foo_Ts_bar) {
    scale_sum += log(foo_T_bar.scale());
  }
  return RxSO3<Scalar>(exp(scale_sum / N),
                       SO3<Scalar>(details::averageUnitQuaternion(foo_Ts_bar)));
}

template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename SequenceContainer::value_type, SE2<Scalar>>::value,
    optional<typename SequenceContainer::value_type>>
average(SequenceContainer const& foo_Ts_bar, int max_num_iterations = 20) {
  // TODO: Implement Proposition 12 from Sec. 6.2 of
  // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
  return iterativeMean(foo_Ts_bar, max_num_iterations);
}

template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename SequenceContainer::value_type, SE3<Scalar>>::value,
    optional<typename SequenceContainer::value_type>>
average(SequenceContainer const& foo_Ts_bar, int max_num_iterations = 20) {
  return iterativeMean(foo_Ts_bar, max_num_iterations);
}

template <class SequenceContainer,
          class Scalar = typename SequenceContainer::value_type::Scalar>
std::enable_if_t<
    std::is_same<typename SequenceContainer::value_type, Sim3<Scalar>>::value,
    optional<typename SequenceContainer::value_type>>
average(SequenceContainer const& foo_Ts_bar, int max_num_iterations = 20) {
  return iterativeMean(foo_Ts_bar, max_num_iterations);
}

}  // namespace Sophus

#endif  // SOPHUS_MEAN_H
