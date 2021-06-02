#pragma once

#include "types.hpp"

namespace Sophus {

template <class Scalar>
class BasisFunction {
 public:
  static SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 4> C() {
    Eigen::Matrix<Scalar, 3, 4> C;
    Scalar const o(0);

    // clang-format off
  C << Scalar(5./6),  Scalar(3./6), -Scalar(3./6),  Scalar(1./6),
       Scalar(1./6),  Scalar(3./6),  Scalar(3./6), -Scalar(2./6),
                  o,             o,             o,  Scalar(1./6);
    // clang-format on
    return C;
  }

  static SOPHUS_FUNC Vector3<Scalar> B(Scalar const& u) {
    // SOPHUS_ENSURE(u >= Scalar(0), "but %", u);
    // SOPHUS_ENSURE(u < Scalar(1), "but %", u);
    Scalar u_square(u * u);
    return C() * Vector4<Scalar>(Scalar(1), u, u_square, u * u_square);
  }

  static SOPHUS_FUNC Vector3<Scalar> Dt_B(Scalar const& u,
                                          Scalar const& delta_t) {
    // SOPHUS_ENSURE(u >= Scalar(0), "but %", u);
    // SOPHUS_ENSURE(u < Scalar(1), "but %", u);
    return (Scalar(1) / delta_t) * C() *
           Vector4<Scalar>(Scalar(0), Scalar(1), Scalar(2) * u,
                           Scalar(3) * u * u);
  }

  static SOPHUS_FUNC Vector3<Scalar> Dt2_B(Scalar const& u,
                                           Scalar const& delta_t) {
    // SOPHUS_ENSURE(u >= Scalar(0), "but %", u);
    // SOPHUS_ENSURE(u < Scalar(1), "but %", u);
    return (Scalar(1) / (delta_t * delta_t)) * C() *
           Vector4<Scalar>(Scalar(0), Scalar(0), Scalar(2), Scalar(6) * u);
  }
};

template <class LieGroup_>
class SplineImpl {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  SplineImpl(std::vector<LieGroup> T_foo_controlPoint, double delta_t)
      : T_foo_controlPoint_(std::move(T_foo_controlPoint)), delta_t_(delta_t) {
    SOPHUS_ENSURE(T_foo_controlPoint_.size() >= 4u, ", but {}",
                  T_foo_controlPoint_.size());

    for (size_t i = 1; i < T_foo_controlPoint_.size(); ++i) {
      control_tagent_vectors_.push_back(
          (T_foo_controlPoint_[i - 1].inverse() * T_foo_controlPoint_[i])
              .log());
    }
  }

  LieGroup T_foo_spline(int i, double u) {
    auto AA = A(i, u);
    return T_foo_controlPoint_.at(i - 1) * std::get<0>(AA) * std::get<1>(AA) *
           std::get<2>(AA);
  }

  Transformation Dt_T_foo_spline(int i, double u) {
    auto AA = A(i, u);
    auto Dt_AA = Dt_A(AA, i, u);
    return T_foo_controlPoint_.at(i - 1).matrix() *
           ((std::get<0>(Dt_AA) * std::get<1>(AA).matrix() *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(Dt_AA) *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(AA).matrix() *
             std::get<2>(Dt_AA)));
  }

  Transformation Dt2_T_foo_spline(int i, double u) {
    auto AA = A(i, u);
    auto Dt_AA = Dt_A(AA, i, u);
    auto Dt2_AA = Dt2_A(AA, Dt_AA, i, u);

    return T_foo_controlPoint_.at(i - 1).matrix() *
           ((std::get<0>(Dt2_AA) * std::get<1>(AA).matrix() *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(Dt2_AA) *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(AA).matrix() *
             std::get<2>(Dt2_AA)) +
            2 * ((std::get<0>(Dt_AA) * std::get<1>(Dt_AA) *
                  std::get<2>(AA).matrix()) +
                 (std::get<0>(Dt_AA) * std::get<1>(AA).matrix() *
                  std::get<2>(Dt_AA)) +
                 (std::get<0>(AA).matrix() * std::get<1>(Dt_AA) *
                  std::get<2>(Dt_AA))));
  }

  double delta_t() const { return delta_t_; }

 private:
  std::tuple<LieGroup, LieGroup, LieGroup> A(int i, double u) {
    Eigen::Vector3d B = BasisFunction<double>::B(u);
    return std::make_tuple(
        LieGroup::exp(B[0] * control_tagent_vectors_.at(i - 1)),
        LieGroup::exp(B[1] * control_tagent_vectors_.at(i)),
        LieGroup::exp(B[2] * control_tagent_vectors_.at(i + 1)));
  }

  std::tuple<Transformation, Transformation, Transformation> Dt_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA, int i, double u) {
    Eigen::Vector3d Dt_B = BasisFunction<double>::Dt_B(u, delta_t_);
    return std::make_tuple(
        Dt_B[0] * std::get<0>(AA).matrix() *
            LieGroup::hat(control_tagent_vectors_.at(i - 1)),
        Dt_B[1] * std::get<1>(AA).matrix() *
            LieGroup::hat(control_tagent_vectors_.at(i)),
        Dt_B[2] * std::get<2>(AA).matrix() *
            LieGroup::hat(control_tagent_vectors_.at(i + 1)));
  }

  std::tuple<Transformation, Transformation, Transformation> Dt2_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA,
      std::tuple<Transformation, Transformation, Transformation> const& Dt_AA,
      int i, double u) {
    Eigen::Vector3d Dt_B = BasisFunction<double>::Dt_B(u, delta_t_);
    Eigen::Vector3d Dt2_B = BasisFunction<double>::Dt2_B(u, delta_t_);

    return std::make_tuple(
        (Dt_B[0] * std::get<0>(Dt_AA).matrix() +
         Dt2_B[0] * std::get<0>(AA).matrix()) *
            LieGroup::hat(control_tagent_vectors_.at(i - 1)),
        (Dt_B[1] * std::get<1>(Dt_AA).matrix() +
         Dt2_B[1] * std::get<1>(AA).matrix()) *
            LieGroup::hat(control_tagent_vectors_.at(i)),
        (Dt_B[2] * std::get<2>(Dt_AA).matrix() +
         Dt2_B[2] * std::get<2>(AA).matrix()) *
            LieGroup::hat(control_tagent_vectors_.at(i + 1)));
  }

  std::vector<LieGroup> T_foo_controlPoint_;
  std::vector<Tangent> control_tagent_vectors_;
  double delta_t_;
};

template <class LieGroup_>
class Spline {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  Spline(std::vector<LieGroup> T_foo_controlPoint, double t0, double delta_t)
      : impl_(std::move(T_foo_controlPoint), delta_t), t0_(t0) {}

  LieGroup T_foo_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.T_foo_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt_T_foo_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt_T_foo_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt2_T_foo_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt2_T_foo_spline(index_and_u.i, index_and_u.u);
  }

  double t0() { return t0_; }

 private:
  struct IndexAndU {
    int i;
    double u;
  };

  double s(double t) const { return (t - t0_) / impl_.delta_t(); }

  IndexAndU index_and_u(double t) const {
    double s = this->s(t);
    double i;
    IndexAndU index_and_u;
    index_and_u.u = std::modf(s, &i);
    index_and_u.i = int(i) + 1;
    return index_and_u;
  }

  SplineImpl<LieGroup> impl_;
  double t0_;
};

}  // namespace Sophus
