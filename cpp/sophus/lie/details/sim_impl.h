// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/core/types.h"

namespace sophus {
namespace details {

template <class ScalarT, int kMatrixDim>
Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> calcW(
    Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> const &omega,
    ScalarT const theta,
    ScalarT const sigma) {
  using std::abs;
  using std::cos;
  using std::exp;
  using std::sin;
  static Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> const kI =
      Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim>::Identity();
  static ScalarT const kOne(1);
  static ScalarT const kHalf(0.5);
  Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> const omega2 = omega * omega;
  ScalarT const scale = exp(sigma);
  ScalarT a;

  ScalarT b;

  ScalarT c;
  if (abs(sigma) < kEpsilon<ScalarT>) {
    c = kOne;
    if (abs(theta) < kEpsilon<ScalarT>) {
      a = kHalf;
      b = ScalarT(1. / 6.);
    } else {
      ScalarT theta_sq = theta * theta;
      a = (kOne - cos(theta)) / theta_sq;
      b = (theta - sin(theta)) / (theta_sq * theta);
    }
  } else {
    c = (scale - kOne) / sigma;
    if (abs(theta) < kEpsilon<ScalarT>) {
      ScalarT sigma_sq = sigma * sigma;
      a = ((sigma - kOne) * scale + kOne) / sigma_sq;
      b = (scale * kHalf * sigma_sq + scale - kOne - sigma * scale) /
          (sigma_sq * sigma);
    } else {
      ScalarT theta_sq = theta * theta;
      ScalarT tmp_a = scale * sin(theta);
      ScalarT tmp_b = scale * cos(theta);
      ScalarT tmp_c = theta_sq + sigma * sigma;
      a = (tmp_a * sigma + (kOne - tmp_b) * theta) / (theta * tmp_c);
      b = (c - ((tmp_b - kOne) * sigma + tmp_a * theta) / (tmp_c)) * kOne /
          (theta_sq);
    }
  }
  return a * omega + b * omega2 + c * kI;
}

template <class ScalarT>
void calcWDerivatives(
    ScalarT const theta,
    ScalarT const sigma,
    ScalarT &a_out,
    ScalarT &b_out,
    ScalarT &c_out,
    ScalarT &a_dsigma_out,
    ScalarT &b_dsigma_out,
    ScalarT &c_dsigma_out,
    ScalarT &a_dtheta_out,
    ScalarT &b_dtheta_out) {
  using std::abs;
  using std::cos;
  using std::exp;
  using std::sin;
  using Scalar = ScalarT;
  static Scalar const kZero(0.0);
  static Scalar const kOne(1.0);
  static Scalar const kHalf(0.5);
  static Scalar const kTwo(2.0);
  static Scalar const kThree(3.0);
  Scalar const theta_sq = theta * theta;
  Scalar const theta_c = theta * theta_sq;
  Scalar const sin_theta = sin(theta);
  Scalar const cos_theta = cos(theta);

  Scalar const scale = exp(sigma);
  Scalar const sigma_sq = sigma * sigma;
  Scalar const sigma_c = sigma * sigma_sq;

  if (abs(sigma) < kEpsilon<ScalarT>) {
    c_out = kOne;
    c_dsigma_out = kHalf;
    if (abs(theta) < kEpsilon<ScalarT>) {
      a_out = kHalf;
      b_out = ScalarT(1. / 6.);
      a_dtheta_out = a_dsigma_out = kZero;
      b_dtheta_out = b_dsigma_out = kZero;
    } else {
      a_out = (kOne - cos_theta) / theta_sq;
      b_out = (theta - sin_theta) / theta_c;
      a_dtheta_out = (theta * sin_theta + kTwo * cos_theta - kTwo) / theta_c;
      b_dtheta_out = -(kTwo * theta - kThree * sin_theta + theta * cos_theta) /
                     (theta_c * theta);
      a_dsigma_out = (sin_theta - theta * cos_theta) / theta_c;
      b_dsigma_out =
          (kHalf - (cos_theta + theta * sin_theta - kOne) / theta_sq) /
          theta_sq;
    }
  } else {
    c_out = (scale - kOne) / sigma;
    c_dsigma_out = (scale * (sigma - kOne) + kOne) / sigma_sq;
    if (abs(theta) < kEpsilon<ScalarT>) {
      a_out = ((sigma - kOne) * scale + kOne) / sigma_sq;
      b_out =
          (scale * kHalf * sigma_sq + scale - kOne - sigma * scale) / sigma_c;
      a_dsigma_out =
          (scale * (sigma_sq - kTwo * sigma + kTwo) - kTwo) / sigma_c;
      b_dsigma_out = (scale * (kHalf * sigma_c - (kOne + kHalf) * sigma_sq +
                               kThree * sigma - kThree) +
                      kThree) /
                     (sigma_c * sigma);
      a_dtheta_out = b_dtheta_out = kZero;
    } else {
      Scalar const a = scale * sin_theta;
      Scalar const b = scale * cos_theta;
      Scalar const b_one = b - kOne;
      Scalar const theta_b_one = theta * b_one;
      Scalar const c = theta_sq + sigma_sq;
      Scalar const c_sq = c * c;
      Scalar const theta_sq_c = theta_sq * c;
      Scalar const a_theta = theta * a;
      Scalar const b_theta = theta * b;
      Scalar const c_theta = theta * c;
      Scalar const a_sigma = sigma * a;
      Scalar const b_sigma = sigma * b;
      Scalar const two_sigma = sigma * kTwo;
      Scalar const two_theta = theta * kTwo;
      Scalar const sigma_b_one = sigma * b_one;

      a_out = (a_sigma - theta_b_one) / c_theta;
      a_dtheta_out = (kTwo * (theta_b_one - a_sigma)) / c_sq +
                     (b_sigma - b + a_theta + kOne) / c_theta +
                     (theta_b_one - a_sigma) / theta_sq_c;
      a_dsigma_out = (a - b_theta + a_sigma) / c_theta -
                     (two_sigma * (theta - b_theta + a_sigma)) / (theta * c_sq);

      b_out = (c_out - (sigma_b_one + a_theta) / (c)) * kOne / (theta_sq);
      b_dtheta_out =
          ((two_theta * (b_sigma - sigma + a_theta)) / c_sq -
           ((a + b_theta - a_sigma)) / c) /
              theta_sq -
          (kTwo * ((scale - kOne) / sigma - (b_sigma - sigma + a_theta) / c)) /
              theta_c;
      b_dsigma_out =
          -((b_sigma + a_theta + b_one) / c + (scale - kOne) / sigma_sq -
            (two_sigma * (sigma_b_one + a_theta)) / c_sq - scale / sigma) /
          theta_sq;
    }
  }
}

template <class ScalarT, int kMatrixDim>
Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> calcWInv(
    Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> const &omega,
    ScalarT const theta,
    ScalarT const sigma,
    ScalarT const scale) {
  using std::abs;
  using std::cos;
  using std::sin;
  static Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> const kI =
      Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim>::Identity();
  static ScalarT const kHalf(0.5);
  static ScalarT const kOne(1);
  static ScalarT const kTwo(2);
  Eigen::Matrix<ScalarT, kMatrixDim, kMatrixDim> const omega2 = omega * omega;
  ScalarT const scale_sq = scale * scale;
  ScalarT const theta_sq = theta * theta;
  ScalarT const sin_theta = sin(theta);
  ScalarT const cos_theta = cos(theta);

  ScalarT a;

  ScalarT b;

  ScalarT c;
  if (abs(sigma * sigma) < kEpsilon<ScalarT>) {
    c = kOne - kHalf * sigma;
    a = -kHalf;
    if (abs(theta_sq) < kEpsilon<ScalarT>) {
      b = ScalarT(1. / 12.);
    } else {
      b = (theta * sin_theta + kTwo * cos_theta - kTwo) /
          (kTwo * theta_sq * (cos_theta - kOne));
    }
  } else {
    ScalarT const scale_cu = scale_sq * scale;
    c = sigma / (scale - kOne);
    if (abs(theta_sq) < kEpsilon<ScalarT>) {
      a = (-sigma * scale + scale - kOne) / ((scale - kOne) * (scale - kOne));
      b = (scale_sq * sigma - kTwo * scale_sq + scale * sigma + kTwo * scale) /
          (kTwo * scale_cu - ScalarT(6) * scale_sq + ScalarT(6) * scale - kTwo);
    } else {
      ScalarT const s_sin_theta = scale * sin_theta;
      ScalarT const s_cos_theta = scale * cos_theta;
      a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
          (theta * (scale_sq - kTwo * s_cos_theta + kOne));
      b = -scale *
          (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
           scale * sigma + sigma * cos_theta - sigma) /
          (theta_sq * (scale_cu - kTwo * scale * s_cos_theta - scale_sq +
                       kTwo * s_cos_theta + scale - kOne));
    }
  }
  return a * omega + b * omega2 + c * kI;
}

}  // namespace details
}  // namespace sophus
