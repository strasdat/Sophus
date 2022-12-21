// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/calculus/num_diff.h"
#include "sophus/common/test_macros.h"
#include "sophus/lie/interp/average.h"
#include "sophus/lie/interp/interpolate.h"
#include "sophus/lie/interp/spline.h"

#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>

#include <array>

#ifdef SOPHUS_CERES
#include <ceres/jet.h>
#endif

namespace sophus {

// compatibility with ceres::Jet types
#if SOPHUS_CERES
using ceres::isfinite;
#else
using std::isfinite;
#endif

template <class TScalar>
Eigen::Hyperplane<TScalar, 2> through(Eigen::Vector<TScalar, 2> const* points) {
  return Eigen::Hyperplane<TScalar, 2>::Through(points[0], points[1]);
}

template <class TScalar>
Eigen::Hyperplane<TScalar, 3> through(Eigen::Vector<TScalar, 3> const* points) {
  return Eigen::Hyperplane<TScalar, 3>::Through(
      points[0], points[1], points[2]);
}

template <class TLieGroup>
class LieGroupTests {
 public:
  using LieGroup = TLieGroup;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;
  using Point = typename LieGroup::Point;
  using HomogeneousPoint = typename LieGroup::HomogeneousPoint;
  using ConstPointMap = Eigen::Map<const Point>;
  using Line = typename LieGroup::Line;
  using Hyperplane = typename LieGroup::Hyperplane;
  using Adjoint = typename LieGroup::Adjoint;
  static int constexpr kPointDim = LieGroup::kPointDim;
  static int constexpr kMatrixDim = LieGroup::kMatrixDim;
  static int constexpr kDoF = LieGroup::kDoF;
  static int constexpr kNumParameters = LieGroup::kNumParameters;

  LieGroupTests(
      std::vector<LieGroup, Eigen::aligned_allocator<LieGroup>> const&
          group_vec,
      std::vector<Tangent, Eigen::aligned_allocator<Tangent>> const&
          tangent_vec,
      std::vector<Point, Eigen::aligned_allocator<Point>> const& point_vec)
      : group_vec_(group_vec),
        tangent_vec_(tangent_vec),
        point_vec_(point_vec) {}

  bool adjointTest() {
    bool passed = true;
    for (size_t i = 0; i < group_vec_.size(); ++i) {
      Transformation t = group_vec_[i].matrix();
      Adjoint ad = group_vec_[i].adj();
      for (size_t j = 0; j < tangent_vec_.size(); ++j) {
        Tangent x = tangent_vec_[j];

        Transformation mat_i;
        mat_i.setIdentity();
        Tangent ad1 = ad * x;
        Tangent ad2 = LieGroup::vee(
            t * LieGroup::hat(x) * group_vec_[i].inverse().matrix());
        SOPHUS_TEST_APPROX(
            passed,
            ad1,
            ad2,
            Scalar(10) * small_eps,
            "Adjoint case %, %",
            mat_i,
            j);
      }
    }
    return passed;
  }

  // For the time being, leftJacobian and leftJacobianInverse are only
  // implemented for So3 and Se3
  template <class TG = LieGroup>
  std::enable_if_t<
      std::is_same<TG, So3<Scalar>>::value ||
          std::is_same<TG, Se3<Scalar>>::value,
      bool>
  leftJacobianTest() {
    bool passed = true;
    for (auto const& x : tangent_vec_) {
      LieGroup const inv_exp_x = LieGroup::exp(x).inverse();

      // Explicit implement the derivative in the Lie Group in first principles
      // as a vector field: D_x f(x) = D_h log(f(x + h) . f(x)^{-1})
      Eigen::Matrix<Scalar, kDoF, kDoF> const j_num =
          vectorFieldNumDiff<Scalar, kDoF, kDoF>(
              [&inv_exp_x](Tangent const& x_plus_delta) {
                return (LieGroup::exp(x_plus_delta) * inv_exp_x).log();
              },
              x);

      // Analytical left Jacobian
      Eigen::Matrix<Scalar, kDoF, kDoF> const j = LieGroup::leftJacobian(x);
      SOPHUS_TEST_APPROX(
          passed, j, j_num, Scalar(100) * small_eps_sqrt, "Left Jacobian");

      Eigen::Matrix<Scalar, kDoF, kDoF> j_inv =
          LieGroup::leftJacobianInverse(x);

      SOPHUS_TEST_APPROX(
          passed,
          j,
          j_inv.inverse().eval(),
          Scalar(100) * small_eps_sqrt,
          "Left Jacobian and its analytical Inverse");
    }

    return passed;
  }

  template <class TG = LieGroup>
  std::enable_if_t<
      !(std::is_same<TG, So3<Scalar>>::value ||
        std::is_same<TG, Se3<Scalar>>::value),
      bool>
  leftJacobianTest() {
    return true;
  }

  bool moreJacobiansTest() {
    bool passed = true;
    for (auto const& point : point_vec_) {
      Eigen::Matrix<Scalar, kPointDim, kDoF> j =
          LieGroup::dxExpXTimesPointAt0(point);
      Tangent t;
      setToZero(t);
      Eigen::Matrix<Scalar, kPointDim, kDoF> const j_num =
          vectorFieldNumDiff<Scalar, kPointDim, kDoF>(
              [point](Tangent const& x) { return LieGroup::exp(x) * point; },
              t);

      SOPHUS_TEST_APPROX(
          passed, j, j_num, small_eps_sqrt, "Dx_exp_x_times_point_at_0");
    }
    return passed;
  }

  bool contructorAndAssignmentTest() {
    bool passed = true;
    for (LieGroup foo_transform_bar : group_vec_) {
      LieGroup foo_t2_bar = foo_transform_bar;
      SOPHUS_TEST_APPROX(
          passed,
          foo_transform_bar.matrix(),
          foo_t2_bar.matrix(),
          small_eps,
          "Copy constructor: %\nvs\n %",
          transpose(foo_transform_bar.matrix()),
          transpose(foo_t2_bar.matrix()));
      LieGroup foo_t3_bar;
      foo_t3_bar = foo_transform_bar;
      SOPHUS_TEST_APPROX(
          passed,
          foo_transform_bar.matrix(),
          foo_t3_bar.matrix(),
          small_eps,
          "Copy assignment: %\nvs\n %",
          transpose(foo_transform_bar.matrix()),
          transpose(foo_t3_bar.matrix()));

      LieGroup foo_t4_bar(foo_transform_bar.matrix());
      SOPHUS_TEST_APPROX(
          passed,
          foo_transform_bar.matrix(),
          foo_t4_bar.matrix(),
          small_eps,
          "Constructor from homogeneous matrix: %\nvs\n %",
          transpose(foo_transform_bar.matrix()),
          transpose(foo_t4_bar.matrix()));

      Eigen::Map<LieGroup> foo_tmap_bar(foo_transform_bar.data());
      LieGroup foo_t5_bar = foo_tmap_bar;
      SOPHUS_TEST_APPROX(
          passed,
          foo_transform_bar.matrix(),
          foo_t5_bar.matrix(),
          small_eps,
          "Assignment from Eigen::Map type: %\nvs\n %",
          transpose(foo_transform_bar.matrix()),
          transpose(foo_t5_bar.matrix()));

      Eigen::Map<LieGroup const> foo_tcmap_bar(foo_transform_bar.data());
      LieGroup foo_t6_bar;
      foo_t6_bar = foo_tcmap_bar;
      SOPHUS_TEST_APPROX(
          passed,
          foo_transform_bar.matrix(),
          foo_t5_bar.matrix(),
          small_eps,
          "Assignment from Eigen::Map type: %\nvs\n %",
          transpose(foo_transform_bar.matrix()),
          transpose(foo_t5_bar.matrix()));

      LieGroup i;
      Eigen::Map<LieGroup> foo_tmap2_bar(i.data());
      foo_tmap2_bar = foo_transform_bar;
      SOPHUS_TEST_APPROX(
          passed,
          foo_tmap2_bar.matrix(),
          foo_transform_bar.matrix(),
          small_eps,
          "Assignment to Eigen::Map type: %\nvs\n %",
          transpose(foo_tmap2_bar.matrix()),
          transpose(foo_transform_bar.matrix()));
    }
    return passed;
  }

  bool derivativeTest() {
    bool passed = true;

    LieGroup g;
    for (int i = 0; i < kDoF; ++i) {
      Transformation gi = g.dxiExpmatXAt0(i);
      Transformation gi2 = curveNumDiff(
          [i](Scalar xi) -> Transformation {
            Tangent x;
            setToZero(x);
            setElementAt(x, xi, i);
            return LieGroup::exp(x).matrix();
          },
          Scalar(0));
      SOPHUS_TEST_APPROX(
          passed, gi, gi2, small_eps_sqrt, "Dxi_exp_x_matrix_at_ case %", i);
    }

    return passed;
  }

  template <class TG = LieGroup>
  bool additionalDerivativeTest() {
    bool passed = true;
    for (size_t j = 0; j < tangent_vec_.size(); ++j) {
      Tangent a = tangent_vec_[j];
      Eigen::Matrix<Scalar, kNumParameters, kDoF> d = LieGroup::dxExpX(a);
      Eigen::Matrix<Scalar, kNumParameters, kDoF> jac_num =
          vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
              [](Tangent const& x) -> Eigen::Vector<Scalar, kNumParameters> {
                return LieGroup::exp(x).params();
              },
              a);

      SOPHUS_TEST_APPROX(
          passed, d, jac_num, 3 * small_eps_sqrt, "dxExpX case: %", j);
    }

    Tangent o;
    setToZero(o);
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j = LieGroup::dxExpXAt0();
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j_num =
        vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
            [](Tangent const& x) -> Eigen::Vector<Scalar, kNumParameters> {
              return LieGroup::exp(x).params();
            },
            o);
    SOPHUS_TEST_APPROX(passed, j, j_num, small_eps_sqrt, "Dx_exp_x_at_0");

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      LieGroup t = group_vec_[i];

      Eigen::Matrix<Scalar, kNumParameters, kDoF> j = t.dxThisMulExpXAt0();
      Eigen::Matrix<Scalar, kNumParameters, kDoF> j_num =
          vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
              [t](Tangent const& x) -> Eigen::Vector<Scalar, kNumParameters> {
                return (t * LieGroup::exp(x)).params();
              },
              o);

      SOPHUS_TEST_APPROX(
          passed,
          j,
          j_num,
          small_eps_sqrt,
          "Dx_this_mul_exp_x_at_0 case: %",
          i);
    }

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      LieGroup t = group_vec_[i];

      Eigen::Matrix<Scalar, kDoF, kDoF> j =
          t.dxLogThisInvTimesXAtThis() * t.dxThisMulExpXAt0();
      Eigen::Matrix<Scalar, kDoF, kDoF> j_exp =
          Eigen::Matrix<Scalar, kDoF, kDoF>::Identity();

      SOPHUS_TEST_APPROX(
          passed,
          j,
          j_exp,
          small_eps_sqrt,
          "Dy_log_this_inv_by_at_x case: %",
          i);
    }
    return passed;
  }

  bool productTest() {
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size() - 1; ++i) {
      LieGroup t1 = group_vec_[i];
      LieGroup t2 = group_vec_[i + 1];
      LieGroup mult = t1 * t2;
      t1 *= t2;
      SOPHUS_TEST_APPROX(
          passed, t1.matrix(), mult.matrix(), small_eps, "Product case: %", i);
    }
    return passed;
  }

  bool expLogTest() {
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      Transformation t1 = group_vec_[i].matrix();
      Transformation t2 = LieGroup::exp(group_vec_[i].log()).matrix();
      SOPHUS_TEST_APPROX(
          passed, t1, t2, small_eps, "G - exp(log(G)) case: %", i);
    }
    return passed;
  }

  bool expMapTest() {
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      Tangent omega = tangent_vec_[i];
      Transformation exp_x = LieGroup::exp(omega).matrix();
      Transformation expmap_hat_x = (LieGroup::hat(omega)).exp();
      SOPHUS_TEST_APPROX(
          passed,
          exp_x,
          expmap_hat_x,
          Scalar(10) * small_eps,
          "expmap(hat(x)) - exp(x) case: %",
          i);
    }
    return passed;
  }

  bool groupActionTest() {
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      for (size_t j = 0; j < point_vec_.size(); ++j) {
        Point const& p = point_vec_[j];
        Point point1 = group_vec_[i] * p;

        HomogeneousPoint hp = p.homogeneous();
        HomogeneousPoint hpoint1 = group_vec_[i] * hp;

        ConstPointMap p_map(p.data());
        Point pointmap1 = group_vec_[i] * p_map;

        Transformation t = group_vec_[i].matrix();
        Point gt_point1 = map(t, p);

        SOPHUS_TEST_APPROX(
            passed, point1, gt_point1, small_eps, "Transform point case: %", i);
        SOPHUS_TEST_APPROX(
            passed,
            hpoint1.hnormalized().eval(),
            gt_point1,
            small_eps,
            "Transform homogeneous point case: %",
            i);
        SOPHUS_TEST_APPROX(
            passed,
            pointmap1,
            gt_point1,
            small_eps,
            "Transform map point case: %",
            i);
      }
    }
    return passed;
  }

  bool lineActionTest() {
    bool passed = point_vec_.size() > 1;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      for (size_t j = 0; j + 1 < point_vec_.size(); ++j) {
        Point const& p1 = point_vec_[j];
        Point const& p2 = point_vec_[j + 1];
        Line l = Line::Through(p1, p2);
        Point p1_t = group_vec_[i] * p1;
        Point p2_t = group_vec_[i] * p2;
        Line l_t = group_vec_[i] * l;

        SOPHUS_TEST_APPROX(
            passed,
            l_t.squaredDistance(p1_t),
            static_cast<Scalar>(0),
            small_eps,
            "Transform line case (1st point) : %",
            i);
        SOPHUS_TEST_APPROX(
            passed,
            l_t.squaredDistance(p2_t),
            static_cast<Scalar>(0),
            small_eps,
            "Transform line case (2nd point) : %",
            i);
        SOPHUS_TEST_APPROX(
            passed,
            l_t.direction().squaredNorm(),
            l.direction().squaredNorm(),
            small_eps,
            "Transform line case (direction) : %",
            i);
      }
    }
    return passed;
  }

  bool planeActionTest() {
    int const point_dim = Point::RowsAtCompileTime;
    bool passed = point_vec_.size() >= point_dim;
    for (size_t i = 0; i < group_vec_.size(); ++i) {
      for (size_t j = 0; j + point_dim - 1 < point_vec_.size(); ++j) {
        Point points[point_dim];

        Point points_t[point_dim];
        for (int k = 0; k < point_dim; ++k) {
          points[k] = point_vec_[j + k];
          points_t[k] = group_vec_[i] * points[k];
        }

        Hyperplane const plane = through(points);

        Hyperplane const plane_t = group_vec_[i] * plane;

        for (int k = 0; k < point_dim; ++k) {
          SOPHUS_TEST_APPROX(
              passed,
              plane_t.signedDistance(points_t[k]),
              static_cast<Scalar>(0.),
              small_eps,
              "Transform plane case (point #%): %",
              k,
              i);
        }
        SOPHUS_TEST_APPROX(
            passed,
            plane_t.normal().squaredNorm(),
            plane.normal().squaredNorm(),
            small_eps,
            "Transform plane case (normal): %",
            i);
      }
    }
    return passed;
  }

  bool lieBracketTest() {
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      for (size_t j = 0; j < tangent_vec_.size(); ++j) {
        Tangent tangent1 =
            LieGroup::lieBracket(tangent_vec_[i], tangent_vec_[j]);
        Transformation hati = LieGroup::hat(tangent_vec_[i]);
        Transformation hatj = LieGroup::hat(tangent_vec_[j]);

        Tangent tangent2 = LieGroup::vee(hati * hatj - hatj * hati);
        SOPHUS_TEST_APPROX(
            passed, tangent1, tangent2, small_eps, "Lie Bracket case: %", i);
      }
    }
    return passed;
  }

  bool veeHatTest() {
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      SOPHUS_TEST_APPROX(
          passed,
          Tangent(tangent_vec_[i]),
          LieGroup::vee(LieGroup::hat(tangent_vec_[i])),
          small_eps,
          "Hat-vee case: %",
          i);
    }
    return passed;
  }

  bool newDeleteSmokeTest() {
    bool passed = true;
    LieGroup* raw_ptr = nullptr;
    raw_ptr = new LieGroup();
    SOPHUS_TEST_NEQ(passed, reinterpret_cast<std::uintptr_t>(raw_ptr), 0, "");
    delete raw_ptr;
    return passed;
  }

  bool interpolateAndMeanTest() {
    bool passed = true;
    using std::sqrt;
    Scalar const eps = kEpsilon<Scalar>;
    Scalar const sqrt_eps = sqrt(eps);
    // TODO: Improve accuracy of ``interpolate`` (and hence ``exp`` and ``log``)
    //       so that we can use more accurate bounds in these tests, i.e.
    //       ``eps`` instead of ``sqrt_eps``.

    for (LieGroup const& foo_transform_bar : group_vec_) {
      for (LieGroup const& foo_transform_daz : group_vec_) {
        // Test boundary conditions ``alpha=0`` and ``alpha=1``.
        LieGroup foo_t_quiz =
            interpolate(foo_transform_bar, foo_transform_daz, Scalar(0));
        SOPHUS_TEST_APPROX(
            passed,
            foo_t_quiz.matrix(),
            foo_transform_bar.matrix(),
            sqrt_eps,
            "");
        foo_t_quiz =
            interpolate(foo_transform_bar, foo_transform_daz, Scalar(1));
        SOPHUS_TEST_APPROX(
            passed,
            foo_t_quiz.matrix(),
            foo_transform_daz.matrix(),
            sqrt_eps,
            "");
      }
    }
    for (Scalar alpha :
         {Scalar(0.1), Scalar(0.5), Scalar(0.75), Scalar(0.99)}) {
      for (LieGroup const& foo_transform_bar : group_vec_) {
        for (LieGroup const& foo_transform_daz : group_vec_) {
          LieGroup foo_t_quiz =
              interpolate(foo_transform_bar, foo_transform_daz, alpha);
          // test left-invariance:
          //
          // dash_T_foo * interp(foo_transform_bar, foo_transform_daz)
          // == interp(dash_T_foo * foo_transform_bar, dash_T_foo *
          // foo_transform_daz)

          if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
                  foo_transform_bar.inverse() * foo_transform_daz)) {
            // skip check since there is a shortest path ambiguity
            continue;
          }
          for (LieGroup const& dash_t_foo : group_vec_) {
            LieGroup dash_t_quiz = interpolate(
                dash_t_foo * foo_transform_bar,
                dash_t_foo * foo_transform_daz,
                alpha);
            SOPHUS_TEST_APPROX(
                passed,
                dash_t_quiz.matrix(),
                (dash_t_foo * foo_t_quiz).matrix(),
                sqrt_eps,
                "");
          }
          // test inverse-invariance:
          //
          // interp(foo_transform_bar, foo_transform_daz).inverse()
          // == interp(foo_transform_bar.inverse(), dash_T_foo.inverse())
          LieGroup quiz_t_foo = interpolate(
              foo_transform_bar.inverse(), foo_transform_daz.inverse(), alpha);
          SOPHUS_TEST_APPROX(
              passed,
              quiz_t_foo.inverse().matrix(),
              foo_t_quiz.matrix(),
              sqrt_eps,
              "");
        }
      }

      for (LieGroup const& bar_transform_foo : group_vec_) {
        for (LieGroup const& baz_transform_foo : group_vec_) {
          LieGroup quiz_t_foo =
              interpolate(bar_transform_foo, baz_transform_foo, alpha);
          // test right-invariance:
          //
          // interp(bar_transform_foo, bar_transform_foo) * foo_T_dash
          // == interp(bar_transform_foo * foo_T_dash, bar_transform_foo *
          // foo_T_dash)

          if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
                  bar_transform_foo * baz_transform_foo.inverse())) {
            // skip check since there is a shortest path ambiguity
            continue;
          }
          for (LieGroup const& foo_t_dash : group_vec_) {
            LieGroup quiz_t_dash = interpolate(
                bar_transform_foo * foo_t_dash,
                baz_transform_foo * foo_t_dash,
                alpha);
            SOPHUS_TEST_APPROX(
                passed,
                quiz_t_dash.matrix(),
                (quiz_t_foo * foo_t_dash).matrix(),
                sqrt_eps,
                "");
          }
        }
      }
    }

    for (LieGroup const& foo_transform_bar : group_vec_) {
      for (LieGroup const& foo_transform_daz : group_vec_) {
        if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
                foo_transform_bar.inverse() * foo_transform_daz)) {
          // skip check since there is a shortest path ambiguity
          continue;
        }

        // test average({A, B}) == interp(A, B):
        LieGroup foo_t_quiz =
            interpolate(foo_transform_bar, foo_transform_daz, 0.5);
        std::optional<LieGroup> foo_t_iaverage = iterativeMean(
            std::array<LieGroup, 2>({{foo_transform_bar, foo_transform_daz}}),
            20);
        std::optional<LieGroup> foo_t_average = average(
            std::array<LieGroup, 2>({{foo_transform_bar, foo_transform_daz}}));
        SOPHUS_TEST(
            passed,
            bool(foo_t_average),
            "log(foo_transform_bar): %\nlog(foo_transform_daz): %",
            transpose(foo_transform_bar.log()),
            transpose(foo_transform_daz.log()),
            "");
        if (foo_t_average) {
          SOPHUS_TEST_APPROX(
              passed,
              foo_t_quiz.matrix(),
              foo_t_average->matrix(),
              sqrt_eps,
              "log(foo_transform_bar): %\nlog(foo_transform_daz): %\n"
              "log(interp): %\nlog(average): %",
              transpose(foo_transform_bar.log()),
              transpose(foo_transform_daz.log()),
              transpose(foo_t_quiz.log()),
              transpose(foo_t_average->log()),
              "");
        }
        SOPHUS_TEST(
            passed,
            bool(foo_t_iaverage),
            "log(foo_transform_bar): %\nlog(foo_transform_daz): %\n"
            "log(interp): %\nlog(iaverage): %",
            transpose(foo_transform_bar.log()),
            transpose(foo_transform_daz.log()),
            transpose(foo_t_quiz.log()),
            transpose(foo_t_iaverage->log()),
            "");
        if (foo_t_iaverage) {
          SOPHUS_TEST_APPROX(
              passed,
              foo_t_quiz.matrix(),
              foo_t_iaverage->matrix(),
              sqrt_eps,
              "log(foo_transform_bar): %\nlog(foo_transform_daz): %",
              transpose(foo_transform_bar.log()),
              transpose(foo_transform_daz.log()),
              "");
        }
      }
    }

    return passed;
  }

  bool testRandomSmoke() {
    bool passed = true;
    std::default_random_engine engine;
    for (int i = 0; i < 100; ++i) {
      LieGroup g = LieGroup::sampleUniform(engine);
      SOPHUS_TEST_EQUAL(passed, g.params(), g.params(), "");
    }
    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_same<TS, float>::value, bool> testSpline() {
    // skip tests for Scalar == float
    return true;
  }

  template <class TS = Scalar>
  std::enable_if_t<!std::is_same<TS, float>::value, bool> testSpline() {
    // run tests for Scalar != float
    bool passed = true;

    for (LieGroup const& t_world_foo : group_vec_) {
      for (LieGroup const& t_world_bar : group_vec_) {
        std::vector<LieGroup> control_poses;
        control_poses.push_back(interpolate(t_world_foo, t_world_bar, 0.0));

        for (double p = 0.2; p < 1.0; p += 0.2) {
          LieGroup t_world_inter = interpolate(t_world_foo, t_world_bar, p);
          control_poses.push_back(t_world_inter);
        }

        BasisSplineImpl<LieGroup> spline(control_poses, 1.0);

        LieGroup t = spline.parentFromSpline(0.0, 1.0);
        LieGroup t2 = spline.parentFromSpline(1.0, 0.0);

        SOPHUS_TEST_APPROX(
            passed,
            t.matrix(),
            t2.matrix(),
            10 * small_eps_sqrt,
            "parent_T_spline");

        Transformation dt_parent_t_spline = spline.dtParentFromSpline(0.0, 0.5);
        Transformation dt_parent_t_spline2 = curveNumDiff(
            [&](double u_bar) -> Transformation {
              return spline.parentFromSpline(0.0, u_bar).matrix();
            },
            0.5);
        SOPHUS_TEST_APPROX(
            passed,
            dt_parent_t_spline,
            dt_parent_t_spline2,
            100 * small_eps_sqrt,
            "Dt_parent_T_spline");

        Transformation dt2_parent_t_spline =
            spline.dt2ParentFromSpline(0.0, 0.5);
        Transformation dt2_parent_t_spline2 = curveNumDiff(
            [&](double u_bar) -> Transformation {
              return spline.dtParentFromSpline(0.0, u_bar).matrix();
            },
            0.5);
        SOPHUS_TEST_APPROX(
            passed,
            dt2_parent_t_spline,
            dt2_parent_t_spline2,
            20 * small_eps_sqrt,
            "Dt2_parent_T_spline");

        for (double frac : {0.01, 0.25, 0.5, 0.9, 0.99}) {
          double t0 = 1.0;
          double delta_t = 0.1;
          BasisSpline<LieGroup> spline(control_poses, t0, delta_t);
          double t = t0 + frac * delta_t;

          Transformation dt_parent_t_spline = spline.dtParentFromSpline(t);
          Transformation dt_parent_t_spline2 = curveNumDiff(
              [&](double t_bar) -> Transformation {
                return spline.parentFromSpline(t_bar).matrix();
              },
              t);
          SOPHUS_TEST_APPROX(
              passed,
              dt_parent_t_spline,
              dt_parent_t_spline2,
              80 * small_eps_sqrt,
              "Dt_parent_T_spline");

          Transformation dt2_parent_t_spline = spline.dt2ParentFromSpline(t);
          Transformation dt2_parent_t_spline2 = curveNumDiff(
              [&](double t_bar) -> Transformation {
                return spline.dtParentFromSpline(t_bar).matrix();
              },
              t);
          SOPHUS_TEST_APPROX(
              passed,
              dt2_parent_t_spline,
              dt2_parent_t_spline2,
              20 * small_eps_sqrt,
              "Dt2_parent_T_spline");
        }
      }
    }
    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_floating_point<TS>::value, bool> doAllTestsPass() {
    return doesLargeTestSetPass();
  }

  template <class TS = Scalar>
  std::enable_if_t<!std::is_floating_point<TS>::value, bool> doAllTestsPass() {
    return doesSmallTestSetPass();
  }

 private:
  bool doesSmallTestSetPass() {
    bool passed = true;
    passed &= adjointTest();
    passed &= contructorAndAssignmentTest();
    passed &= productTest();
    passed &= expLogTest();
    passed &= groupActionTest();
    passed &= lineActionTest();
    passed &= planeActionTest();
    passed &= lieBracketTest();
    passed &= veeHatTest();
    passed &= newDeleteSmokeTest();
    return passed;
  }

  bool doesLargeTestSetPass() {
    bool passed = true;
    passed &= doesSmallTestSetPass();
    passed &= additionalDerivativeTest();
    passed &= derivativeTest();
    passed &= expMapTest();
    passed &= interpolateAndMeanTest();
    passed &= testRandomSmoke();
    passed &= testSpline();
    passed &= leftJacobianTest();
    passed &= moreJacobiansTest();
    return passed;
  }

  Scalar const small_eps = kEpsilon<Scalar>;
  Scalar const small_eps_sqrt = kEpsilonSqrt<Scalar>;

  Eigen::Matrix<Scalar, kMatrixDim - 1, 1> map(
      Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim> const& t,
      Eigen::Matrix<Scalar, kMatrixDim - 1, 1> const& p) {
    return t.template topLeftCorner<kMatrixDim - 1, kMatrixDim - 1>() * p +
           t.template topRightCorner<kMatrixDim - 1, 1>();
  }

  Eigen::Matrix<Scalar, kMatrixDim, 1> map(
      Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim> const& t,
      Eigen::Matrix<Scalar, kMatrixDim, 1> const& p) {
    return t * p;
  }

  std::vector<LieGroup, Eigen::aligned_allocator<LieGroup>> group_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

template <class TScalar>
std::vector<Se3<TScalar>, Eigen::aligned_allocator<Se3<TScalar>>>
getTestSE3s() {
  TScalar const k_pi = kPi<TScalar>;
  std::vector<Se3<TScalar>, Eigen::aligned_allocator<Se3<TScalar>>> se3_vec;
  se3_vec.push_back(Se3<TScalar>(
      So3<TScalar>::exp(
          Eigen::Vector3<TScalar>(TScalar(0.2), TScalar(0.5), TScalar(0.0))),
      Eigen::Vector3<TScalar>(TScalar(0), TScalar(0), TScalar(0))));
  se3_vec.push_back(Se3<TScalar>(
      So3<TScalar>::exp(
          Eigen::Vector3<TScalar>(TScalar(0.2), TScalar(0.5), TScalar(-1.0))),
      Eigen::Vector3<TScalar>(TScalar(10), TScalar(0), TScalar(0))));
  se3_vec.push_back(Se3<TScalar>::trans(
      Eigen::Vector3<TScalar>(TScalar(0), TScalar(100), TScalar(5))));
  se3_vec.push_back(Se3<TScalar>::rotZ(TScalar(0.00001)));
  se3_vec.push_back(
      Se3<TScalar>::trans(
          TScalar(0), TScalar(-0.00000001), TScalar(0.0000000001)) *
      Se3<TScalar>::rotZ(TScalar(0.00001)));
  se3_vec.push_back(
      Se3<TScalar>::transX(TScalar(0.01)) *
      Se3<TScalar>::rotZ(TScalar(0.00001)));
  se3_vec.push_back(
      Se3<TScalar>::trans(TScalar(4), TScalar(-5), TScalar(0)) *
      Se3<TScalar>::rotX(k_pi));
  se3_vec.push_back(
      Se3<TScalar>(
          So3<TScalar>::exp(Eigen::Vector3<TScalar>(
              TScalar(0.2), TScalar(0.5), TScalar(0.0))),
          Eigen::Vector3<TScalar>(TScalar(0), TScalar(0), TScalar(0))) *
      Se3<TScalar>::rotX(k_pi) *
      Se3<TScalar>(
          So3<TScalar>::exp(Eigen::Vector3<TScalar>(
              TScalar(-0.2), TScalar(-0.5), TScalar(-0.0))),
          Eigen::Vector3<TScalar>(TScalar(0), TScalar(0), TScalar(0))));
  se3_vec.push_back(
      Se3<TScalar>(
          So3<TScalar>::exp(Eigen::Vector3<TScalar>(
              TScalar(0.3), TScalar(0.5), TScalar(0.1))),
          Eigen::Vector3<TScalar>(TScalar(2), TScalar(0), TScalar(-7))) *
      Se3<TScalar>::rotX(k_pi) *
      Se3<TScalar>(
          So3<TScalar>::exp(Eigen::Vector3<TScalar>(
              TScalar(-0.3), TScalar(-0.5), TScalar(-0.1))),
          Eigen::Vector3<TScalar>(TScalar(0), TScalar(6), TScalar(0))));
  return se3_vec;
}

template <class TT>
std::vector<Se2<TT>, Eigen::aligned_allocator<Se2<TT>>> getTestSE2s() {
  std::vector<Se2<TT>, Eigen::aligned_allocator<Se2<TT>>> se2_vec;
  se2_vec.push_back(Se2<TT>());
  se2_vec.push_back(Se2<TT>(So2<TT>(0.2), Eigen::Vector2<TT>(10, 0)));
  se2_vec.push_back(Se2<TT>::transY(100));
  se2_vec.push_back(Se2<TT>::trans(Eigen::Vector2<TT>(1, 2)));
  se2_vec.push_back(Se2<TT>(So2<TT>(-1.), Eigen::Vector2<TT>(20, -1)));
  se2_vec.push_back(
      Se2<TT>(So2<TT>(0.00001), Eigen::Vector2<TT>(-0.00000001, 0.0000000001)));
  se2_vec.push_back(
      Se2<TT>(So2<TT>(0.3), Eigen::Vector2<TT>(2, 0)) * Se2<TT>::rot(kPi<TT>) *
      Se2<TT>(So2<TT>(-0.3), Eigen::Vector2<TT>(0, 6)));
  return se2_vec;
}
}  // namespace sophus
