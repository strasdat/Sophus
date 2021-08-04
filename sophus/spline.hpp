#pragma once

#include "types.hpp"

namespace Sophus {

template <class Scalar>

// C-spline implementation on Lie Group following:
// S. Lovegrove, A. Patron-Perez, G. Sibley, BMVC 2013
// http://www.bmva.org/bmvc/2013/Papers/paper0093/paper0093.pdf
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

  static LieGroup parent_T_spline(
      const LieGroup& parent_Ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u) {
    auto AA = A(control_tagent_vectors, u);
    return parent_Ts_control_point * std::get<0>(AA) * std::get<1>(AA) *
           std::get<2>(AA);
  }

  static Transformation Dt_parent_T_spline(
      const LieGroup& parent_Ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u, double delta_t) {
    auto AA = A(control_tagent_vectors, u);
    auto Dt_AA = Dt_A(AA, control_tagent_vectors, u, delta_t);
    return parent_Ts_control_point.matrix() *
           ((std::get<0>(Dt_AA) * std::get<1>(AA).matrix() *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(Dt_AA) *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(AA).matrix() *
             std::get<2>(Dt_AA)));
  }

  static Transformation Dt2_parent_T_spline(
      const LieGroup& parent_Ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u, double delta_t) {
    auto AA = A(control_tagent_vectors, u);
    auto Dt_AA = Dt_A(AA, control_tagent_vectors, u, delta_t);
    auto Dt2_AA = Dt2_A(AA, Dt_AA, control_tagent_vectors, u, delta_t);

    return parent_Ts_control_point.matrix() *
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
            LieGroup::hat(std::get<1>(control_tagent_vectors)),
        Dt_B[2] * std::get<2>(AA).matrix() *
            LieGroup::hat(std::get<2>(control_tagent_vectors)));
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
};

template <class LieGroup_>
class SplineImpl {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  SplineImpl(std::vector<LieGroup> parent_Ts_control_point, double delta_t)
      : parent_Ts_control_point_(parent_Ts_control_point), delta_t_(delta_t) {
    SOPHUS_ENSURE(parent_Ts_control_point_.size() >= 2u, ", but {}",
                  parent_Ts_control_point_.size());
    recomputeControlTangentVectors();
    SOPHUS_ENSURE(
        parent_Ts_control_point_.size() + 1 == control_tagent_vectors_.size(),
        "{} vs {}", parent_Ts_control_point_.size(),
        control_tagent_vectors_.size());
  }

  LieGroup parent_T_spline(int i, double u) {
    SOPHUS_ENSURE(i >= 0, "i = {}", i);
    SOPHUS_ENSURE(
        i < this->getNumSegments(),
        "i = {};  this->getNumSegments() = {}; control_tagent_vectors_.size() "
        "= {}; parent_Ts_control_point_.size() = {}",
        i, this->getNumSegments(), control_tagent_vectors_.size(),
        parent_Ts_control_point_.size());

    return SplineFn<LieGroup>::parent_T_spline(
        get_parent_T_control_point(i),
        std::make_tuple(control_tagent_vectors_[i],
                        control_tagent_vectors_[i + 1],
                        control_tagent_vectors_[i + 2]),
        u);
  }

  Transformation Dt_parent_T_spline(int i, double u) {
    SOPHUS_ENSURE(i >= 0, "i = {}", i);
    SOPHUS_ENSURE(
        i < this->getNumSegments(),
        "i = {};  this->getNumSegments() = {}; control_tagent_vectors_.size() "
        "= {}; parent_Ts_control_point_.size() = {}",
        i, this->getNumSegments(), control_tagent_vectors_.size(),
        parent_Ts_control_point_.size());

    return SplineFn<LieGroup>::Dt_parent_T_spline(
        get_parent_T_control_point(i),
        std::make_tuple(control_tagent_vectors_[i],
                        control_tagent_vectors_[i + 1],
                        control_tagent_vectors_[i + 2]),
        u, delta_t_);
  }

  Transformation Dt2_parent_T_spline(int i, double u) {
    SOPHUS_ENSURE(i >= 0, "i = {}", i);
    SOPHUS_ENSURE(
        i < this->getNumSegments(),
        "i = {};  this->getNumSegments() = {}; control_tagent_vectors_.size() "
        "= {}; parent_Ts_control_point_.size() = {}",
        i, this->getNumSegments(), control_tagent_vectors_.size(),
        parent_Ts_control_point_.size());

    return SplineFn<LieGroup>::Dt2_parent_T_spline(
        get_parent_T_control_point(i),
        std::make_tuple(control_tagent_vectors_[i],
                        control_tagent_vectors_[i + 1],
                        control_tagent_vectors_[i + 2]),
        u, delta_t_);
  }

  const std::vector<LieGroup>& parent_Ts_control_point() const {
    return parent_Ts_control_point_;
  }

  std::vector<LieGroup>& parent_Ts_control_point() {
    return parent_Ts_control_point_;
  }

  int getNumSegments() const {
    return int(parent_Ts_control_point_.size()) - 1;
  }

  double delta_t() const { return delta_t_; }

  void recomputeControlTangentVectors() {
    control_tagent_vectors_.clear();
    Tangent o;
    setToZero(o);
    control_tagent_vectors_.push_back(o);
    for (size_t i = 1; i < parent_Ts_control_point_.size(); ++i) {
      control_tagent_vectors_.push_back(
          (parent_Ts_control_point_[i - 1].inverse() *
           parent_Ts_control_point_[i])
              .log());
    }
    control_tagent_vectors_.push_back(o);
  }

 private:
  // The first two, and the last two control poses are equal:
  //  - get_parent_T_control_point(0) == get_parent_T_control_point(1)
  //  - get_parent_T_control_point(s-1) == get_parent_T_control_point(s-2)
  // Hence we do not store the first and last pose, but rather deal with the
  // corner cases here.
  //
  // Hence, the first and last contol tangent vectors are 0. See
  // `recomputeControlTangentVectors`.
  LieGroup const& get_parent_T_control_point(int i) const {
    if (i == 0) {
      return parent_Ts_control_point_.front();
    }
    if (i == int(parent_Ts_control_point_.size())) {
      return parent_Ts_control_point_.back();
    }
    return parent_Ts_control_point_[i - 1];
  }

  std::vector<LieGroup> parent_Ts_control_point_;
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

  Spline(std::vector<LieGroup> parent_Ts_control_point, double t0,
         double delta_t)
      : impl_(std::move(parent_Ts_control_point), delta_t), t0_(t0) {}

  LieGroup parent_T_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);

    return impl_.parent_T_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt_parent_T_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt_parent_T_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt2_parent_T_spline(double t) {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt2_parent_T_spline(index_and_u.i, index_and_u.u);
  }

  double t0() const { return t0_; }

  double tmax() const { return t0_ + impl_.delta_t() * getNumSegments(); }

  const std::vector<LieGroup>& parent_Ts_control_point() const {
    return impl_.parent_Ts_control_point();
  }

  std::vector<LieGroup>& parent_Ts_control_point() {
    return impl_.parent_Ts_control_point();
  }

  void recomputeControlTangentVectors() {
    impl_.recomputeControlTangentVectors();
  }

  int getNumSegments() const { return impl_.getNumSegments(); }

 private:
  struct IndexAndU {
    int i;
    double u;
  };

  double s(double t) const { return (t - t0_) / impl_.delta_t(); }

  IndexAndU index_and_u(double t) const {
    SOPHUS_ENSURE(t >= t0_, "{} vs. {}", t, t0_);
    SOPHUS_ENSURE(t <= this->tmax(), "{} vs. {}", t, this->tmax());

    double s = this->s(t);
    double i;
    IndexAndU index_and_u;
    index_and_u.u = std::modf(s, &i);
    index_and_u.i = int(i);
    if (index_and_u.u > 0.0) {
      return index_and_u;
    }

    // u == 0.0
    if (s < 0.5 * this->tmax()) {
      // First half of spline, keep as is (i, 0.0).
      return index_and_u;
    }
    // Second half of spline, use (i-1, 1.0) instead. This way we can represent
    // t == tmax (and not just t<tmax).
    index_and_u.u = 1.0;
    --index_and_u.i;

    return index_and_u;
  }

  SplineImpl<LieGroup> impl_;

  double t0_;
};

}  // namespace Sophus
