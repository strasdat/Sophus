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
class SplineFn {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  // SplineImpl(std::vector<LieGroup> T_foo_controlPoint, double delta_t)
  // // T_foo_controlPoint_(std::move(T_foo_controlPoint))
  // {
  //   // SOPHUS_ENSURE(T_foo_controlPoint_.size() >= 4u, ", but {}",
  //   //               T_foo_controlPoint_.size());

  //   // for (size_t i = 1; i < T_foo_controlPoint_.size(); ++i) {
  //   //   control_tagent_vectors_.push_back(
  //   //       (T_foo_controlPoint_[i - 1].inverse() * T_foo_controlPoint_[i])
  //   //           .log());
  //   // }
  // }

  // in i-1
  static LieGroup T_foo_spline(
      const LieGroup& T_foo_controlPoint,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u) {
    auto AA = A(control_tagent_vectors, u);
    return T_foo_controlPoint * std::get<0>(AA) * std::get<1>(AA) *
           std::get<2>(AA);
  }

  static Transformation Dt_T_foo_spline(
      const LieGroup& T_foo_controlPoint,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u, double delta_t) {
    auto AA = A(control_tagent_vectors, u);
    auto Dt_AA = Dt_A(AA, control_tagent_vectors, u, delta_t);
    return T_foo_controlPoint.matrix() *
           ((std::get<0>(Dt_AA) * std::get<1>(AA).matrix() *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(Dt_AA) *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(AA).matrix() *
             std::get<2>(Dt_AA)));
  }

  // in i-1
  static Transformation Dt2_T_foo_spline(
      const LieGroup& T_foo_controlPoint,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u, double delta_t) {
    auto AA = A(control_tagent_vectors, u);
    auto Dt_AA = Dt_A(AA, control_tagent_vectors, u, delta_t);
    auto Dt2_AA = Dt2_A(AA, Dt_AA, control_tagent_vectors, u, delta_t);

    return T_foo_controlPoint.matrix() *
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

 private:
  static std::tuple<LieGroup, LieGroup, LieGroup> A(
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u) {
    Eigen::Vector3d B = BasisFunction<double>::B(u);
    return std::make_tuple(
        LieGroup::exp(B[0] * std::get<0>(control_tagent_vectors)),
        LieGroup::exp(B[1] * std::get<1>(control_tagent_vectors)),
        LieGroup::exp(B[2] * std::get<2>(control_tagent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> Dt_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA,
      const std::tuple<Tangent, Tangent, Tangent>& control_tagent_vectors,
      double u, double delta_t) {
    Eigen::Vector3d Dt_B = BasisFunction<double>::Dt_B(u, delta_t);
    return std::make_tuple(
        Dt_B[0] * std::get<0>(AA).matrix() *
            LieGroup::hat(std::get<0>(control_tagent_vectors)),
        Dt_B[1] * std::get<1>(AA).matrix() *
            LieGroup::hat(std::get<0>(control_tagent_vectors)),
        Dt_B[2] * std::get<2>(AA).matrix() *
            LieGroup::hat(std::get<0>(control_tagent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> Dt2_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA,
      std::tuple<Transformation, Transformation, Transformation> const& Dt_AA,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u, double delta_t) {
    Eigen::Vector3d Dt_B = BasisFunction<double>::Dt_B(u, delta_t);
    Eigen::Vector3d Dt2_B = BasisFunction<double>::Dt2_B(u, delta_t);

    return std::make_tuple(
        (Dt_B[0] * std::get<0>(Dt_AA).matrix() +
         Dt2_B[0] * std::get<0>(AA).matrix()) *
            LieGroup::hat(std::get<0>(control_tagent_vectors)),
        (Dt_B[1] * std::get<1>(Dt_AA).matrix() +
         Dt2_B[1] * std::get<1>(AA).matrix()) *
            LieGroup::hat(std::get<1>(control_tagent_vectors)),
        (Dt_B[2] * std::get<2>(Dt_AA).matrix() +
         Dt2_B[2] * std::get<2>(AA).matrix()) *
            LieGroup::hat(std::get<2>(control_tagent_vectors)));
  }

  // std::vector<LieGroup> T_foo_controlPoint_;
  // std::vector<Tangent> control_tagent_vectors_;
  // double delta_t_;
};

template <class LieGroup_>
class SplineImpl {
};

template <class LieGroup_>
class Spline {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  Spline(std::vector<LieGroup> T_foo_controlPoint, double t0, double delta_t)
      : T_foo_controlPoint_(std::move(T_foo_controlPoint)),
        t0_(t0),
        delta_t_(delta_t) {
    SOPHUS_ENSURE(T_foo_controlPoint_.size() >= 4u, ", but {}",
                  T_foo_controlPoint_.size());

    for (size_t i = 1; i < T_foo_controlPoint_.size(); ++i) {
      control_tagent_vectors_.push_back(
          (T_foo_controlPoint_[i - 1].inverse() * T_foo_controlPoint_[i])
              .log());
    }
  }

  LieGroup T_foo_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);
    size_t i = index_and_u.i;
    //FARM_NG_CHECK_GE(i, 0);
    //FARM_NG_CHECK_LT(i, control_tagent_vectors_.size());

    if (i == 0) {
      return SplineFn<LieGroup>::T_foo_spline(
          T_foo_controlPoint_.at(i),
          std::make_tuple(control_tagent_vectors_[i],
                          control_tagent_vectors_[i],
                          control_tagent_vectors_[i + 1]),
          index_and_u.u);
    }

    if (i == control_tagent_vectors_.size()-1) {
      return SplineFn<LieGroup>::T_foo_spline(
          T_foo_controlPoint_.at(i),
          std::make_tuple(control_tagent_vectors_[i-1],
                          control_tagent_vectors_[i],
                          control_tagent_vectors_[i]),
          index_and_u.u);
    }

    return SplineFn<LieGroup>::T_foo_spline(
        T_foo_controlPoint_.at(i - 1),
        std::make_tuple(control_tagent_vectors_[i - 1],
                        control_tagent_vectors_[i],
                        control_tagent_vectors_[i + 1]),
        index_and_u.u);
  }

  // Transformation Dt_T_foo_spline(double t) {
  //   IndexAndU index_and_u = this->index_and_u(t);
  //   return SplineImpl<LieGroup>::Dt_T_foo_spline(
  //       T_foo_controlPoint_.at(index_and_u.i - 1),
  //       std::make_tuple(control_tagent_vectors_[index_and_u.i - 1],
  //                       control_tagent_vectors_[index_and_u.i],
  //                       control_tagent_vectors_[index_and_u.i + 1]),
  //       index_and_u.u, delta_t_);
  // }

  // Transformation Dt2_T_foo_spline(double t) {
  //   IndexAndU index_and_u = this->index_and_u(t);
  //   return SplineImpl<LieGroup>::Dt2_T_foo_spline(
  //       T_foo_controlPoint_.at(index_and_u.i - 1),
  //       std::make_tuple(control_tagent_vectors_[index_and_u.i - 1],
  //                       control_tagent_vectors_[index_and_u.i],
  //                       control_tagent_vectors_[index_and_u.i + 1]),
  //       index_and_u.u, delta_t_);
  // }

  double t0() { return t0_; }

  const std::vector<LieGroup>& T_foo_controlPoint() const {
    return T_foo_controlPoint_;
  }

  std::vector<LieGroup>& T_foo_controlPoint() { return T_foo_controlPoint_; }

 private:
  struct IndexAndU {
    int i;
    double u;
  };

  double s(double t) const { return (t - t0_) / delta_t_; }

  IndexAndU index_and_u(double t) const {
    double s = this->s(t);
    double i;
    IndexAndU index_and_u;
    index_and_u.u = std::modf(s, &i);
    index_and_u.i = int(i) + 1;
    return index_and_u;
  }

  // SplineImpl<LieGroup> impl_;

  std::vector<LieGroup> T_foo_controlPoint_;
  std::vector<Tangent> control_tagent_vectors_;
  double t0_;
  double delta_t_;
};

}  // namespace Sophus
