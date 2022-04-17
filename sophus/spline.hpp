/// @file
// Basis spline implementation on Lie Group following:
// S. Lovegrove, A. Patron-Perez, G. Sibley, BMVC 2013
// http://www.bmva.org/bmvc/2013/Papers/paper0093/paper0093.pdf

#pragma once

#include "types.hpp"

namespace Sophus {

template <class Scalar>
class SplineBasisFunction {
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
class BasisSplineFn {
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
    using Scalar = typename LieGroup::Scalar;
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
            Scalar(2) * ((std::get<0>(Dt_AA) * std::get<1>(Dt_AA) *
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
    Eigen::Vector3d B = SplineBasisFunction<double>::B(u);
    return std::make_tuple(
        LieGroup::exp(B[0] * std::get<0>(control_tagent_vectors)),
        LieGroup::exp(B[1] * std::get<1>(control_tagent_vectors)),
        LieGroup::exp(B[2] * std::get<2>(control_tagent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> Dt_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA,
      const std::tuple<Tangent, Tangent, Tangent>& control_tagent_vectors,
      double u, double delta_t) {
    Eigen::Vector3d Dt_B = SplineBasisFunction<double>::Dt_B(u, delta_t);
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
    Eigen::Vector3d Dt_B = SplineBasisFunction<double>::Dt_B(u, delta_t);
    Eigen::Vector3d Dt2_B = SplineBasisFunction<double>::Dt2_B(u, delta_t);

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

enum class SegmentCase { first, normal, last };

template <class LieGroup>
struct BasisSplineSegment {
 public:
  using T = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;

  BasisSplineSegment(SegmentCase segment_case, T const* const raw_ptr0,
                     T const* const raw_ptr1, T const* const raw_ptr2,
                     T const* const raw_ptr3)
      : segment_case_(segment_case),
        raw_params0_(raw_ptr0),
        raw_params1_(raw_ptr1),
        raw_params2_(raw_ptr2),
        raw_params3_(raw_ptr3) {}

  Eigen::Map<LieGroup const> const world_pose_foo_prev() const {
    return Eigen::Map<LieGroup const>(raw_params0_);
  }
  Eigen::Map<LieGroup const> const world_pose_foo_0() const {
    return Eigen::Map<LieGroup const>(raw_params1_);
  }

  Eigen::Map<LieGroup const> const world_pose_foo_1() const {
    return Eigen::Map<LieGroup const>(raw_params2_);
  }

  Eigen::Map<LieGroup const> const world_pose_foo_2() const {
    return Eigen::Map<LieGroup const>(raw_params3_);
  }
  LieGroup parent_T_spline(double u) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<LieGroup>::parent_T_spline(
            world_pose_foo_0(),
            std::make_tuple(
                (world_pose_foo_0().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u);
      case SegmentCase::normal:
        return BasisSplineFn<LieGroup>::parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u);
      case SegmentCase::last:
        return BasisSplineFn<LieGroup>::parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_1()).log()),
            u);
    }
    SOPHUS_ENSURE(false, "logic error");
  }

  Transformation Dt_parent_T_spline(double u, double delta_t) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<LieGroup>::Dt_parent_T_spline(
            world_pose_foo_0(),
            std::make_tuple(
                (world_pose_foo_0().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::normal:
        return BasisSplineFn<LieGroup>::Dt_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::last:
        return BasisSplineFn<LieGroup>::Dt_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_1()).log()),
            u, delta_t);
    }
    SOPHUS_ENSURE(false, "logic error");
  }

  Transformation Dt2_parent_T_spline(double u, double delta_t) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<LieGroup>::Dt2_parent_T_spline(
            world_pose_foo_0(),
            std::make_tuple(
                (world_pose_foo_0().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::normal:
        return BasisSplineFn<LieGroup>::Dt2_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::last:
        return BasisSplineFn<LieGroup>::Dt2_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_1()).log()),
            u, delta_t);
    }
    SOPHUS_ENSURE(false, "logic error");
  }

 private:
  SegmentCase segment_case_;
  T const* raw_params0_;
  T const* raw_params1_;
  T const* raw_params2_;
  T const* raw_params3_;
};

template <class LieGroup_>
class BasisSplineImpl {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  BasisSplineImpl(const std::vector<LieGroup>& parent_Ts_control_point,
                  double delta_t)
      : parent_Ts_control_point_(parent_Ts_control_point), delta_t_(delta_t) {
    SOPHUS_ENSURE(parent_Ts_control_point_.size() >= 2u, ", but {}",
                  parent_Ts_control_point_.size());
  }

  LieGroup parent_T_spline(int i, double u) const {
    SOPHUS_ENSURE(i >= 0, "i = {}", i);
    SOPHUS_ENSURE(i < this->getNumSegments(),
                  "i = {};  this->getNumSegments() = {};  "
                  "parent_Ts_control_point_.size() = {}",
                  i, this->getNumSegments(), parent_Ts_control_point_.size());

    SegmentCase segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    int idx_prev = std::max(0, i - 1);
    int idx_0 = i;
    int idx_1 = i + 1;
    int idx_2 = std::min(i + 2, int(this->parent_Ts_control_point_.size()) - 1);

    return BasisSplineSegment<LieGroup>(
               segment_case, parent_Ts_control_point_[idx_prev].data(),
               parent_Ts_control_point_[idx_0].data(),
               parent_Ts_control_point_[idx_1].data(),
               parent_Ts_control_point_[idx_2].data())
        .parent_T_spline(u);
  }

  Transformation Dt_parent_T_spline(int i, double u) const {
    SOPHUS_ENSURE(i >= 0, "i = {}", i);
    SOPHUS_ENSURE(i < this->getNumSegments(),
                  "i = {};  this->getNumSegments() = {};  "
                  "parent_Ts_control_point_.size() = {}",
                  i, this->getNumSegments(), parent_Ts_control_point_.size());

    SegmentCase segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    int idx_prev = std::max(0, i - 1);
    int idx_0 = i;
    int idx_1 = i + 1;
    int idx_2 = std::min(i + 2, int(this->parent_Ts_control_point_.size()) - 1);

    return BasisSplineSegment<LieGroup>(
               segment_case, parent_Ts_control_point_[idx_prev].data(),
               parent_Ts_control_point_[idx_0].data(),
               parent_Ts_control_point_[idx_1].data(),
               parent_Ts_control_point_[idx_2].data())
        .Dt_parent_T_spline(u, delta_t_);
  }

  Transformation Dt2_parent_T_spline(int i, double u) const {
    SOPHUS_ENSURE(i >= 0, "i = {}", i);
    SOPHUS_ENSURE(i < this->getNumSegments(),
                  "i = {};  this->getNumSegments() = {};  "
                  "parent_Ts_control_point_.size() = {}",
                  i, this->getNumSegments(), parent_Ts_control_point_.size());

    SegmentCase segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    int idx_prev = std::max(0, i - 1);
    int idx_0 = i;
    int idx_1 = i + 1;
    int idx_2 = std::min(i + 2, int(this->parent_Ts_control_point_.size()) - 1);

    return BasisSplineSegment<LieGroup>(
               segment_case, parent_Ts_control_point_[idx_prev].data(),
               parent_Ts_control_point_[idx_0].data(),
               parent_Ts_control_point_[idx_1].data(),
               parent_Ts_control_point_[idx_2].data())
        .Dt2_parent_T_spline(u, delta_t_);
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

 private:
  std::vector<LieGroup> parent_Ts_control_point_;
  double delta_t_;
};

struct IndexAndU {
  int i;
  double u;
};

template <class LieGroup_>
class BasisSpline {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  BasisSpline(std::vector<LieGroup> parent_Ts_control_point, double t0,
              double delta_t)
      : impl_(std::move(parent_Ts_control_point), delta_t), t0_(t0) {}

  LieGroup parent_T_spline(double t) const {
    IndexAndU index_and_u = this->index_and_u(t);

    return impl_.parent_T_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt_parent_T_spline(double t) const {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt_parent_T_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt2_parent_T_spline(double t) const {
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

  int getNumSegments() const { return impl_.getNumSegments(); }

  double s(double t) const { return (t - t0_) / impl_.delta_t(); }

  double delta_t() const { return impl_.delta_t(); }

  IndexAndU index_and_u(double t) const {
    SOPHUS_ENSURE(t >= t0_, "{} vs. {}", t, t0_);
    SOPHUS_ENSURE(t <= this->tmax(), "{} vs. {}", t, this->tmax());

    double s = this->s(t);
    double i;
    IndexAndU index_and_u;
    index_and_u.u = std::modf(s, &i);
    index_and_u.i = int(i);
    if (index_and_u.u > Sophus::Constants<double>::epsilon()) {
      return index_and_u;
    }

    // u ~=~ 0.0
    if (t < 0.5 * this->tmax()) {
      // First half of spline, keep as is (i, 0.0).
      return index_and_u;
    }
    // Second half of spline, use (i-1, 1.0) instead. This way we can represent
    // t == tmax (and not just t<tmax).
    index_and_u.u += 1.0;
    --index_and_u.i;

    return index_and_u;
  }

 private:
  BasisSplineImpl<LieGroup> impl_;

  double t0_;
};

}  // namespace Sophus
