// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
// Basis spline implementation on Lie Group following:
// S. Lovegrove, A. Patron-Perez, G. Sibley, BMVC 2013
// http://www.bmva.org/bmvc/2013/Papers/paper0093/paper0093.pdf

#pragma once

#include "sophus/common/types.h"

#include <cmath>

namespace sophus {

template <class TScalar>
class SplineBasisFunction {
 public:
  static Eigen::Matrix<TScalar, 3, 4> c() {
    Eigen::Matrix<TScalar, 3, 4> c;
    TScalar const o(0);

    // clang-format off
    c << TScalar(5./6),  TScalar(3./6), -TScalar(3./6),  TScalar(1./6),
         TScalar(1./6),  TScalar(3./6),  TScalar(3./6), -TScalar(2./6),
                    o,             o,             o,  TScalar(1./6);
    // clang-format on
    return c;
  }

  static Eigen::Vector3<TScalar> b(TScalar const& u) {
    // SOPHUS_ASSERT(u >= TScalar(0), "but %", u);
    // SOPHUS_ASSERT(u < TScalar(1), "but %", u);
    TScalar u_square(u * u);
    return c() * Eigen::Vector4<TScalar>(TScalar(1), u, u_square, u * u_square);
  }

  static Eigen::Vector3<TScalar> dtB(TScalar const& u, TScalar const& delta_t) {
    // SOPHUS_ASSERT(u >= TScalar(0), "but %", u);
    // SOPHUS_ASSERT(u < TScalar(1), "but %", u);
    return (TScalar(1) / delta_t) * c() *
           Eigen::Vector4<TScalar>(
               TScalar(0), TScalar(1), TScalar(2) * u, TScalar(3) * u * u);
  }

  static Eigen::Vector3<TScalar> dt2B(
      TScalar const& u, TScalar const& delta_t) {
    // SOPHUS_ASSERT(u >= TScalar(0), "but %", u);
    // SOPHUS_ASSERT(u < TScalar(1), "but %", u);
    return (TScalar(1) / (delta_t * delta_t)) * c() *
           Eigen::Vector4<TScalar>(
               TScalar(0), TScalar(0), TScalar(2), TScalar(6) * u);
  }
};

template <class TLieGroup>
class BasisSplineFn {
 public:
  using LieGroup = TLieGroup;
  using TScalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  static LieGroup parentFromSpline(
      LieGroup const& parent_ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u) {
    auto aa = a(control_tagent_vectors, u);
    return parent_ts_control_point * std::get<0>(aa) * std::get<1>(aa) *
           std::get<2>(aa);
  }

  static Transformation dtParentFromSpline(
      LieGroup const& parent_ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u,
      double delta_t) {
    auto aa = a(control_tagent_vectors, u);
    auto dt_aa = dtA(aa, control_tagent_vectors, u, delta_t);
    return parent_ts_control_point.matrix() *
           ((std::get<0>(dt_aa) * std::get<1>(aa).matrix() *
             std::get<2>(aa).matrix()) +
            (std::get<0>(aa).matrix() * std::get<1>(dt_aa) *
             std::get<2>(aa).matrix()) +
            (std::get<0>(aa).matrix() * std::get<1>(aa).matrix() *
             std::get<2>(dt_aa)));
  }

  static Transformation dt2ParentFromSpline(
      LieGroup const& parent_ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u,
      double delta_t) {
    using TScalar = typename LieGroup::Scalar;
    auto aa = a(control_tagent_vectors, u);
    auto dt_aa = dtA(aa, control_tagent_vectors, u, delta_t);
    auto dt2_aa = dt2A(aa, dt_aa, control_tagent_vectors, u, delta_t);

    return parent_ts_control_point.matrix() *
           ((std::get<0>(dt2_aa) * std::get<1>(aa).matrix() *
             std::get<2>(aa).matrix()) +
            (std::get<0>(aa).matrix() * std::get<1>(dt2_aa) *
             std::get<2>(aa).matrix()) +
            (std::get<0>(aa).matrix() * std::get<1>(aa).matrix() *
             std::get<2>(dt2_aa)) +
            TScalar(2) * ((std::get<0>(dt_aa) * std::get<1>(dt_aa) *
                           std::get<2>(aa).matrix()) +
                          (std::get<0>(dt_aa) * std::get<1>(aa).matrix() *
                           std::get<2>(dt_aa)) +
                          (std::get<0>(aa).matrix() * std::get<1>(dt_aa) *
                           std::get<2>(dt_aa))));
  }

 private:
  static std::tuple<LieGroup, LieGroup, LieGroup> a(
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u) {
    Eigen::Vector3d b = SplineBasisFunction<double>::b(u);
    return std::make_tuple(
        LieGroup::exp(b[0] * std::get<0>(control_tagent_vectors)),
        LieGroup::exp(b[1] * std::get<1>(control_tagent_vectors)),
        LieGroup::exp(b[2] * std::get<2>(control_tagent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> dtA(
      std::tuple<LieGroup, LieGroup, LieGroup> const& aa,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u,
      double delta_t) {
    Eigen::Vector3d dt_b = SplineBasisFunction<double>::dtB(u, delta_t);
    return std::make_tuple(
        dt_b[0] * std::get<0>(aa).matrix() *
            LieGroup::hat(std::get<0>(control_tagent_vectors)),
        dt_b[1] * std::get<1>(aa).matrix() *
            LieGroup::hat(std::get<1>(control_tagent_vectors)),
        dt_b[2] * std::get<2>(aa).matrix() *
            LieGroup::hat(std::get<2>(control_tagent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> dt2A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& aa,
      std::tuple<Transformation, Transformation, Transformation> const& dt_aa,
      std::tuple<Tangent, Tangent, Tangent> const& control_tagent_vectors,
      double u,
      double delta_t) {
    Eigen::Vector3d dt_b = SplineBasisFunction<double>::dtB(u, delta_t);
    Eigen::Vector3d dt2_b = SplineBasisFunction<double>::dt2B(u, delta_t);

    return std::make_tuple(
        (dt_b[0] * std::get<0>(dt_aa).matrix() +
         dt2_b[0] * std::get<0>(aa).matrix()) *
            LieGroup::hat(std::get<0>(control_tagent_vectors)),
        (dt_b[1] * std::get<1>(dt_aa).matrix() +
         dt2_b[1] * std::get<1>(aa).matrix()) *
            LieGroup::hat(std::get<1>(control_tagent_vectors)),
        (dt_b[2] * std::get<2>(dt_aa).matrix() +
         dt2_b[2] * std::get<2>(aa).matrix()) *
            LieGroup::hat(std::get<2>(control_tagent_vectors)));
  }
};

enum class SegmentCase { first, normal, last };

template <class TLieGroup>
struct BasisSplineSegment {
 public:
  using T = typename TLieGroup::Scalar;
  using Transformation = typename TLieGroup::Transformation;

  BasisSplineSegment(
      SegmentCase segment_case,
      T const* const raw_ptr0,
      T const* const raw_ptr1,
      T const* const raw_ptr2,
      T const* const raw_ptr3)
      : segment_case_(segment_case),
        raw_params0_(raw_ptr0),
        raw_params1_(raw_ptr1),
        raw_params2_(raw_ptr2),
        raw_params3_(raw_ptr3) {}

  [[nodiscard]] Eigen::Map<TLieGroup const> worldFromFooPrev() const {
    return Eigen::Map<TLieGroup const>(raw_params0_);
  }
  [[nodiscard]] Eigen::Map<TLieGroup const> worldFromFoo0() const {
    return Eigen::Map<TLieGroup const>(raw_params1_);
  }

  [[nodiscard]] Eigen::Map<TLieGroup const> worldFromFoo1() const {
    return Eigen::Map<TLieGroup const>(raw_params2_);
  }

  [[nodiscard]] Eigen::Map<TLieGroup const> worldFromFoo2() const {
    return Eigen::Map<TLieGroup const>(raw_params3_);
  }
  TLieGroup parentFromSpline(double u) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<TLieGroup>::parentFromSpline(
            worldFromFoo0(),
            std::make_tuple(
                (worldFromFoo0().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo2()).log()),
            u);
      case SegmentCase::normal:
        return BasisSplineFn<TLieGroup>::parentFromSpline(
            worldFromFooPrev(),
            std::make_tuple(
                (worldFromFooPrev().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo2()).log()),
            u);
      case SegmentCase::last:
        return BasisSplineFn<TLieGroup>::parentFromSpline(
            worldFromFooPrev(),
            std::make_tuple(
                (worldFromFooPrev().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo1()).log()),
            u);
    }
    SOPHUS_ASSERT(false, "logic error");
  }

  Transformation dtParentFromSpline(double u, double delta_t) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<TLieGroup>::dtParentFromSpline(
            worldFromFoo0(),
            std::make_tuple(
                (worldFromFoo0().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo2()).log()),
            u,
            delta_t);
      case SegmentCase::normal:
        return BasisSplineFn<TLieGroup>::dtParentFromSpline(
            worldFromFooPrev(),
            std::make_tuple(
                (worldFromFooPrev().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo2()).log()),
            u,
            delta_t);
      case SegmentCase::last:
        return BasisSplineFn<TLieGroup>::dtParentFromSpline(
            worldFromFooPrev(),
            std::make_tuple(
                (worldFromFooPrev().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo1()).log()),
            u,
            delta_t);
    }
    SOPHUS_ASSERT(false, "logic error");
  }

  Transformation dt2ParentFromSpline(double u, double delta_t) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<TLieGroup>::dt2ParentFromSpline(
            worldFromFoo0(),
            std::make_tuple(
                (worldFromFoo0().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo2()).log()),
            u,
            delta_t);
      case SegmentCase::normal:
        return BasisSplineFn<TLieGroup>::dt2ParentFromSpline(
            worldFromFooPrev(),
            std::make_tuple(
                (worldFromFooPrev().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo2()).log()),
            u,
            delta_t);
      case SegmentCase::last:
        return BasisSplineFn<TLieGroup>::dt2ParentFromSpline(
            worldFromFooPrev(),
            std::make_tuple(
                (worldFromFooPrev().inverse() * worldFromFoo0()).log(),
                (worldFromFoo0().inverse() * worldFromFoo1()).log(),
                (worldFromFoo1().inverse() * worldFromFoo1()).log()),
            u,
            delta_t);
    }
    SOPHUS_ASSERT(false, "logic error");
  }

 private:
  SegmentCase segment_case_;
  T const* raw_params0_;
  T const* raw_params1_;
  T const* raw_params2_;
  T const* raw_params3_;
};

template <class TLieGroup>
class BasisSplineImpl {
 public:
  using LieGroup = TLieGroup;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  BasisSplineImpl(
      std::vector<LieGroup> const& parent_ts_control_point, double delta_t)
      : parent_from_control_point_transforms_(parent_ts_control_point),
        delta_transform_(delta_t) {
    SOPHUS_ASSERT(
        parent_from_control_point_transforms_.size() >= 2u,
        ", but {}",
        parent_from_control_point_transforms_.size());
  }

  [[nodiscard]] LieGroup parentFromSpline(int i, double u) const {
    SOPHUS_ASSERT(i >= 0, "i = {}", i);
    SOPHUS_ASSERT(
        i < this->getNumSegments(),
        "i = {};  this->getNumSegments() = {};  "
        "parent_from_control_point_transforms_.size() = {}",
        i,
        this->getNumSegments(),
        parent_from_control_point_transforms_.size());

    SegmentCase segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    int idx_prev = std::max(0, i - 1);
    int idx_0 = i;
    int idx_1 = i + 1;
    int idx_2 = std::min(
        i + 2, int(this->parent_from_control_point_transforms_.size()) - 1);

    return BasisSplineSegment<LieGroup>(
               segment_case,
               parent_from_control_point_transforms_[idx_prev].data(),
               parent_from_control_point_transforms_[idx_0].data(),
               parent_from_control_point_transforms_[idx_1].data(),
               parent_from_control_point_transforms_[idx_2].data())
        .parentFromSpline(u);
  }

  [[nodiscard]] Transformation dtParentFromSpline(int i, double u) const {
    SOPHUS_ASSERT(i >= 0, "i = {}", i);
    SOPHUS_ASSERT(
        i < this->getNumSegments(),
        "i = {};  this->getNumSegments() = {};  "
        "parent_from_control_point_transforms_.size() = {}",
        i,
        this->getNumSegments(),
        parent_from_control_point_transforms_.size());

    SegmentCase segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    int idx_prev = std::max(0, i - 1);
    int idx_0 = i;
    int idx_1 = i + 1;
    int idx_2 = std::min(
        i + 2, int(this->parent_from_control_point_transforms_.size()) - 1);

    return BasisSplineSegment<LieGroup>(
               segment_case,
               parent_from_control_point_transforms_[idx_prev].data(),
               parent_from_control_point_transforms_[idx_0].data(),
               parent_from_control_point_transforms_[idx_1].data(),
               parent_from_control_point_transforms_[idx_2].data())
        .dtParentFromSpline(u, delta_transform_);
  }

  [[nodiscard]] Transformation dt2ParentFromSpline(int i, double u) const {
    SOPHUS_ASSERT(i >= 0, "i = {}", i);
    SOPHUS_ASSERT(
        i < this->getNumSegments(),
        "i = {};  this->getNumSegments() = {};  "
        "parent_from_control_point_transforms_.size() = {}",
        i,
        this->getNumSegments(),
        parent_from_control_point_transforms_.size());

    SegmentCase segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    int idx_prev = std::max(0, i - 1);
    int idx_0 = i;
    int idx_1 = i + 1;
    int idx_2 = std::min(
        i + 2, int(this->parent_from_control_point_transforms_.size()) - 1);

    return BasisSplineSegment<LieGroup>(
               segment_case,
               parent_from_control_point_transforms_[idx_prev].data(),
               parent_from_control_point_transforms_[idx_0].data(),
               parent_from_control_point_transforms_[idx_1].data(),
               parent_from_control_point_transforms_[idx_2].data())
        .dt2ParentFromSpline(u, delta_transform_);
  }

  [[nodiscard]] std::vector<LieGroup> const& parentFromsControlPoint() const {
    return parent_from_control_point_transforms_;
  }

  std::vector<LieGroup>& parentFromsControlPoint() {
    return parent_from_control_point_transforms_;
  }

  [[nodiscard]] int getNumSegments() const {
    return int(parent_from_control_point_transforms_.size()) - 1;
  }

  [[nodiscard]] double deltaT() const { return delta_transform_; }

 private:
  std::vector<LieGroup> parent_from_control_point_transforms_;
  double delta_transform_;
};

struct IndexAndU {
  int i;
  double u;
};

template <class TLieGroup>
class BasisSpline {
 public:
  using LieGroup = TLieGroup;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  BasisSpline(
      std::vector<LieGroup> parent_ts_control_point, double t0, double delta_t)
      : impl_(std::move(parent_ts_control_point), delta_t), t0_(t0) {}

  [[nodiscard]] LieGroup parentFromSpline(double t) const {
    IndexAndU index_and_u = this->indexAndU(t);

    return impl_.parentFromSpline(index_and_u.i, index_and_u.u);
  }

  [[nodiscard]] Transformation dtParentFromSpline(double t) const {
    IndexAndU index_and_u = this->indexAndU(t);
    return impl_.dtParentFromSpline(index_and_u.i, index_and_u.u);
  }

  [[nodiscard]] Transformation dt2ParentFromSpline(double t) const {
    IndexAndU index_and_u = this->indexAndU(t);
    return impl_.dt2ParentFromSpline(index_and_u.i, index_and_u.u);
  }

  [[nodiscard]] double t0() const { return t0_; }

  [[nodiscard]] double tmax() const {
    return t0_ + impl_.deltaT() * getNumSegments();
  }

  [[nodiscard]] std::vector<LieGroup> const& parentFromsControlPoint() const {
    return impl_.parentFromsControlPoint();
  }

  std::vector<LieGroup>& parentFromsControlPoint() {
    return impl_.parentFromsControlPoint();
  }

  [[nodiscard]] int getNumSegments() const { return impl_.getNumSegments(); }

  [[nodiscard]] double s(double t) const { return (t - t0_) / impl_.deltaT(); }

  [[nodiscard]] double deltaT() const { return impl_.deltaT(); }

  [[nodiscard]] IndexAndU indexAndU(double t) const {
    SOPHUS_ASSERT(t >= t0_, "{} vs. {}", t, t0_);
    SOPHUS_ASSERT(t <= this->tmax(), "{} vs. {}", t, this->tmax());

    double s = this->s(t);
    double i = NAN;
    IndexAndU index_and_u;
    index_and_u.u = std::modf(s, &i);
    index_and_u.i = int(i);
    if (index_and_u.u > sophus::kEpsilonF64) {
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

}  // namespace sophus
