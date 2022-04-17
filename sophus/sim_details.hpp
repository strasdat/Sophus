#pragma once

#include "types.hpp"

namespace Sophus {
namespace details {

template <class Scalar, int N>
Matrix<Scalar, N, N> calcW(Matrix<Scalar, N, N> const &Omega,
                           Scalar const theta, Scalar const sigma) {
  using std::abs;
  using std::cos;
  using std::exp;
  using std::sin;
  static Matrix<Scalar, N, N> const I = Matrix<Scalar, N, N>::Identity();
  static Scalar const one(1);
  static Scalar const half(0.5);
  Matrix<Scalar, N, N> const Omega2 = Omega * Omega;
  Scalar const scale = exp(sigma);
  Scalar A, B, C;
  if (abs(sigma) < Constants<Scalar>::epsilon()) {
    C = one;
    if (abs(theta) < Constants<Scalar>::epsilon()) {
      A = half;
      B = Scalar(1. / 6.);
    } else {
      Scalar theta_sq = theta * theta;
      A = (one - cos(theta)) / theta_sq;
      B = (theta - sin(theta)) / (theta_sq * theta);
    }
  } else {
    C = (scale - one) / sigma;
    if (abs(theta) < Constants<Scalar>::epsilon()) {
      Scalar sigma_sq = sigma * sigma;
      A = ((sigma - one) * scale + one) / sigma_sq;
      B = (scale * half * sigma_sq + scale - one - sigma * scale) /
          (sigma_sq * sigma);
    } else {
      Scalar theta_sq = theta * theta;
      Scalar a = scale * sin(theta);
      Scalar b = scale * cos(theta);
      Scalar c = theta_sq + sigma * sigma;
      A = (a * sigma + (one - b) * theta) / (theta * c);
      B = (C - ((b - one) * sigma + a * theta) / (c)) * one / (theta_sq);
    }
  }
  return A * Omega + B * Omega2 + C * I;
}

template <class Scalar>
void calcW_derivatives(Scalar const theta, Scalar const sigma, Scalar &A,
                       Scalar &B, Scalar &C, Scalar &A_dsigma, Scalar &B_dsigma,
                       Scalar &C_dsigma, Scalar &A_dtheta, Scalar &B_dtheta) {
  using std::abs;
  using std::cos;
  using std::exp;
  using std::sin;
  static Scalar const zero(0.0);
  static Scalar const one(1.0);
  static Scalar const half(0.5);
  static Scalar const two(2.0);
  static Scalar const three(3.0);
  Scalar const theta_sq = theta * theta;
  Scalar const theta_c = theta * theta_sq;
  Scalar const sin_theta = sin(theta);
  Scalar const cos_theta = cos(theta);

  Scalar const scale = exp(sigma);
  Scalar const sigma_sq = sigma * sigma;
  Scalar const sigma_c = sigma * sigma_sq;

  if (abs(sigma) < Constants<Scalar>::epsilon()) {
    C = one;
    C_dsigma = half;
    if (abs(theta) < Constants<Scalar>::epsilon()) {
      A = half;
      B = Scalar(1. / 6.);
      A_dtheta = A_dsigma = zero;
      B_dtheta = B_dsigma = zero;
    } else {
      A = (one - cos_theta) / theta_sq;
      B = (theta - sin_theta) / theta_c;
      A_dtheta = (theta * sin_theta + two * cos_theta - two) / theta_c;
      B_dtheta = -(two * theta - three * sin_theta + theta * cos_theta) /
                 (theta_c * theta);
      A_dsigma = (sin_theta - theta * cos_theta) / theta_c;
      B_dsigma =
          (half - (cos_theta + theta * sin_theta - one) / theta_sq) / theta_sq;
    }
  } else {
    C = (scale - one) / sigma;
    C_dsigma = (scale * (sigma - one) + one) / sigma_sq;
    if (abs(theta) < Constants<Scalar>::epsilon()) {
      A = ((sigma - one) * scale + one) / sigma_sq;
      B = (scale * half * sigma_sq + scale - one - sigma * scale) / sigma_c;
      A_dsigma = (scale * (sigma_sq - two * sigma + two) - two) / sigma_c;
      B_dsigma = (scale * (half * sigma_c - (one + half) * sigma_sq +
                           three * sigma - three) +
                  three) /
                 (sigma_c * sigma);
      A_dtheta = B_dtheta = zero;
    } else {
      Scalar const a = scale * sin_theta;
      Scalar const b = scale * cos_theta;
      Scalar const b_one = b - one;
      Scalar const theta_b_one = theta * b_one;
      Scalar const c = theta_sq + sigma_sq;
      Scalar const c_sq = c * c;
      Scalar const theta_sq_c = theta_sq * c;
      Scalar const a_theta = theta * a;
      Scalar const b_theta = theta * b;
      Scalar const c_theta = theta * c;
      Scalar const a_sigma = sigma * a;
      Scalar const b_sigma = sigma * b;
      Scalar const two_sigma = sigma * two;
      Scalar const two_theta = theta * two;
      Scalar const sigma_b_one = sigma * b_one;

      A = (a_sigma - theta_b_one) / c_theta;
      A_dtheta = (two * (theta_b_one - a_sigma)) / c_sq +
                 (b_sigma - b + a_theta + one) / c_theta +
                 (theta_b_one - a_sigma) / theta_sq_c;
      A_dsigma = (a - b_theta + a_sigma) / c_theta -
                 (two_sigma * (theta - b_theta + a_sigma)) / (theta * c_sq);

      B = (C - (sigma_b_one + a_theta) / (c)) * one / (theta_sq);
      B_dtheta =
          ((two_theta * (b_sigma - sigma + a_theta)) / c_sq -
           ((a + b_theta - a_sigma)) / c) /
              theta_sq -
          (two * ((scale - one) / sigma - (b_sigma - sigma + a_theta) / c)) /
              theta_c;
      B_dsigma =
          -((b_sigma + a_theta + b_one) / c + (scale - one) / sigma_sq -
            (two_sigma * (sigma_b_one + a_theta)) / c_sq - scale / sigma) /
          theta_sq;
    }
  }
}

template <class Scalar, int N>
Matrix<Scalar, N, N> calcWInv(Matrix<Scalar, N, N> const &Omega,
                              Scalar const theta, Scalar const sigma,
                              Scalar const scale) {
  using std::abs;
  using std::cos;
  using std::sin;
  static Matrix<Scalar, N, N> const I = Matrix<Scalar, N, N>::Identity();
  static Scalar const half(0.5);
  static Scalar const one(1);
  static Scalar const two(2);
  Matrix<Scalar, N, N> const Omega2 = Omega * Omega;
  Scalar const scale_sq = scale * scale;
  Scalar const theta_sq = theta * theta;
  Scalar const sin_theta = sin(theta);
  Scalar const cos_theta = cos(theta);

  Scalar a, b, c;
  if (abs(sigma * sigma) < Constants<Scalar>::epsilon()) {
    c = one - half * sigma;
    a = -half;
    if (abs(theta_sq) < Constants<Scalar>::epsilon()) {
      b = Scalar(1. / 12.);
    } else {
      b = (theta * sin_theta + two * cos_theta - two) /
          (two * theta_sq * (cos_theta - one));
    }
  } else {
    Scalar const scale_cu = scale_sq * scale;
    c = sigma / (scale - one);
    if (abs(theta_sq) < Constants<Scalar>::epsilon()) {
      a = (-sigma * scale + scale - one) / ((scale - one) * (scale - one));
      b = (scale_sq * sigma - two * scale_sq + scale * sigma + two * scale) /
          (two * scale_cu - Scalar(6) * scale_sq + Scalar(6) * scale - two);
    } else {
      Scalar const s_sin_theta = scale * sin_theta;
      Scalar const s_cos_theta = scale * cos_theta;
      a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
          (theta * (scale_sq - two * s_cos_theta + one));
      b = -scale *
          (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
           scale * sigma + sigma * cos_theta - sigma) /
          (theta_sq * (scale_cu - two * scale * s_cos_theta - scale_sq +
                       two * s_cos_theta + scale - one));
    }
  }
  return a * Omega + b * Omega2 + c * I;
}

}  // namespace details
}  // namespace Sophus
