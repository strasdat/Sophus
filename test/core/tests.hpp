#ifndef SOPUHS_TESTS_HPP
#define SOPUHS_TESTS_HPP

#include <array>

#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>

#include <sophus/average.hpp>
#include <sophus/interpolate.hpp>
#include <sophus/test_macros.hpp>

namespace Sophus {

template <class LieGroup>
class LieGroupTests {
 public:
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;
  using Point = typename LieGroup::Point;
  using Line = typename LieGroup::Line;
  using Adjoint = typename LieGroup::Adjoint;
  static int constexpr N = LieGroup::N;
  static int constexpr DoF = LieGroup::DoF;

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
      Transformation T = group_vec_[i].matrix();
      Adjoint Ad = group_vec_[i].Adj();
      for (size_t j = 0; j < tangent_vec_.size(); ++j) {
        Tangent x = tangent_vec_[j];

        Transformation I;
        I.setIdentity();
        Tangent ad1 = Ad * x;
        Tangent ad2 = LieGroup::vee(T * LieGroup::hat(x) *
                                    group_vec_[i].inverse().matrix());
        SOPHUS_TEST_APPROX(passed, ad1, ad2, 20. * kSmallEps,
                           "Adjoint case %, %", i, j);
      }
    }
    return passed;
  }

  bool expLogTest() {
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      Transformation T1 = group_vec_[i].matrix();
      Transformation T2 = LieGroup::exp(group_vec_[i].log()).matrix();
      SOPHUS_TEST_APPROX(passed, T1, T2, kSmallEps, "G - exp(log(G)) case: %",
                         i);
    }
    return passed;
  }

  bool expMapTest() {
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      Tangent omega = tangent_vec_[i];
      Transformation exp_x = LieGroup::exp(omega).matrix();
      Transformation expmap_hat_x = (LieGroup::hat(omega)).exp();
      SOPHUS_TEST_APPROX(passed, exp_x, expmap_hat_x, 10. * kSmallEps,
                         "expmap(hat(x)) - exp(x) case: %", i);
    }
    return passed;
  }

  bool groupActionTest() {
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      for (size_t j = 0; j < point_vec_.size(); ++j) {
        Point const& p = point_vec_[j];
        Transformation T = group_vec_[i].matrix();
        Point point1 = group_vec_[i] * p;
        Point point2 = map(T, p);
        SOPHUS_TEST_APPROX(passed, point1, point2, kSmallEps,
                           "Transform point case: %", i);
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

        SOPHUS_TEST_APPROX(passed, l_t.squaredDistance(p1_t),
                           static_cast<Scalar>(0), kSmallEps,
                           "Transform line case (1st point) : %", i);
        SOPHUS_TEST_APPROX(passed, l_t.squaredDistance(p2_t),
                           static_cast<Scalar>(0), kSmallEps,
                           "Transform line case (2nd point) : %", i);
        SOPHUS_TEST_APPROX(passed, l_t.direction().squaredNorm(),
                           l.direction().squaredNorm(), kSmallEps,
                           "Transform line case (direction) : %", i);
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
        SOPHUS_TEST_APPROX(passed, tangent1, tangent2, kSmallEps,
                           "Lie Bracket case: %", i);
      }
    }
    return passed;
  }

  bool veeHatTest() {
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      SOPHUS_TEST_APPROX(passed, Tangent(tangent_vec_[i]),
                         LieGroup::vee(LieGroup::hat(tangent_vec_[i])),
                         kSmallEps, "Hat-vee case: %", i);
    }
    return passed;
  }

  bool newDeleteSmokeTest() {
    bool passed = true;
    LieGroup* raw_ptr = nullptr;
    raw_ptr = new LieGroup();
    SOPHUS_TEST_NEQ(passed, reinterpret_cast<std::uintptr_t>(raw_ptr), 0);
    delete raw_ptr;
    return passed;
  }

  bool interpolateAndMeanTest() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    Scalar const sqrt_eps = std::sqrt(eps);
    // TODO: Improve accuracy of ``interpolate`` (and hence ``exp`` and ``log``)
    //       so that we can use more accurate bounds in these tests, i.e.
    //       ``eps`` instead of ``sqrt_eps``.

    for (LieGroup const& foo_T_bar : group_vec_) {
      for (LieGroup const& foo_T_baz : group_vec_) {
        // Test boundary conditions ``alpha=0`` and ``alpha=1``.
        LieGroup foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, Scalar(0));
        SOPHUS_TEST_APPROX(passed, foo_T_quiz.matrix(), foo_T_bar.matrix(),
                           sqrt_eps);
        foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, Scalar(1));
        SOPHUS_TEST_APPROX(passed, foo_T_quiz.matrix(), foo_T_baz.matrix(),
                           sqrt_eps);
      }
    }
    for (Scalar alpha : {0.1, 0.5, 0.75, 0.99}) {
      for (LieGroup const& foo_T_bar : group_vec_) {
        for (LieGroup const& foo_T_baz : group_vec_) {
          LieGroup foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, alpha);
          // test left-invariance:
          //
          // dash_T_foo * interp(foo_T_bar, foo_T_baz)
          // == interp(dash_T_foo * foo_T_bar, dash_T_foo * foo_T_baz)

          if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
                  foo_T_bar.inverse() * foo_T_baz)) {
            // skip check since there is a shortest path ambiguity
            continue;
          }
          for (LieGroup const& dash_T_foo : group_vec_) {
            LieGroup dash_T_quiz = interpolate(dash_T_foo * foo_T_bar,
                                               dash_T_foo * foo_T_baz, alpha);
            SOPHUS_TEST_APPROX(passed, dash_T_quiz.matrix(),
                               (dash_T_foo * foo_T_quiz).matrix(), sqrt_eps);
          }
          // test inverse-invariance:
          //
          // interp(foo_T_bar, foo_T_baz).inverse()
          // == interp(foo_T_bar.inverse(), dash_T_foo.inverse())
          LieGroup quiz_T_foo =
              interpolate(foo_T_bar.inverse(), foo_T_baz.inverse(), alpha);
          SOPHUS_TEST_APPROX(passed, quiz_T_foo.inverse().matrix(),
                             foo_T_quiz.matrix(), sqrt_eps);
        }
      }

      for (LieGroup const& bar_T_foo : group_vec_) {
        for (LieGroup const& baz_T_foo : group_vec_) {
          LieGroup quiz_T_foo = interpolate(bar_T_foo, baz_T_foo, alpha);
          // test right-invariance:
          //
          // interp(bar_T_foo, bar_T_foo) * foo_T_dash
          // == interp(bar_T_foo * foo_T_dash, bar_T_foo * foo_T_dash)

          if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
                  bar_T_foo * baz_T_foo.inverse())) {
            // skip check since there is a shortest path ambiguity
            continue;
          }
          for (LieGroup const& foo_T_dash : group_vec_) {
            LieGroup quiz_T_dash = interpolate(bar_T_foo * foo_T_dash,
                                               baz_T_foo * foo_T_dash, alpha);
            SOPHUS_TEST_APPROX(passed, quiz_T_dash.matrix(),
                               (quiz_T_foo * foo_T_dash).matrix(), sqrt_eps);
          }
        }
      }
    }

    for (LieGroup const& foo_T_bar : group_vec_) {
      for (LieGroup const& foo_T_baz : group_vec_) {
        if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
                foo_T_bar.inverse() * foo_T_baz)) {
          // skip check since there is a shortest path ambiguity
          continue;
        }

        // test average({A, B}) == interp(A, B):
        LieGroup foo_T_quiz = interpolate(foo_T_bar, foo_T_baz, 0.5);
        optional<LieGroup> foo_T_iaverage = iterativeMean(
            std::array<LieGroup, 2>({{foo_T_bar, foo_T_baz}}), 20);
        optional<LieGroup> foo_T_average =
            average(std::array<LieGroup, 2>({{foo_T_bar, foo_T_baz}}));
        SOPHUS_TEST(passed, bool(foo_T_average),
                    "log(foo_T_bar): %\nlog(foo_T_baz): %",
                    transpose(foo_T_bar.log()), transpose(foo_T_baz.log()));
        if (foo_T_average) {
          SOPHUS_TEST_APPROX(
              passed, foo_T_quiz.matrix(), foo_T_average->matrix(), sqrt_eps,
              "log(foo_T_bar): %\nlog(foo_T_baz): %\n"
              "log(interp): %\nlog(average): %",
              transpose(foo_T_bar.log()), transpose(foo_T_baz.log()),
              transpose(foo_T_quiz.log()), transpose(foo_T_average->log()));
        }
        SOPHUS_TEST(passed, bool(foo_T_iaverage),
                    "log(foo_T_bar): %\nlog(foo_T_baz): %\n"
                    "log(interp): %\nlog(iaverage): %",
                    transpose(foo_T_bar.log()), transpose(foo_T_baz.log()),
                    transpose(foo_T_quiz.log()),
                    transpose(foo_T_iaverage->log()));
        if (foo_T_iaverage) {
          SOPHUS_TEST_APPROX(
              passed, foo_T_quiz.matrix(), foo_T_iaverage->matrix(), sqrt_eps,
              "log(foo_T_bar): %\nlog(foo_T_baz): %",
              transpose(foo_T_bar.log()), transpose(foo_T_baz.log()));
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
      std::cout << g.matrix() << std::endl << std::endl;
    }
    return passed;
  }

  bool doAllTestsPass() {
    bool passed = true;
    passed &= adjointTest();
    passed &= expLogTest();
    passed &= expMapTest();
    passed &= groupActionTest();
    passed &= lineActionTest();
    passed &= lieBracketTest();
    passed &= veeHatTest();
    passed &= newDeleteSmokeTest();
    passed &= interpolateAndMeanTest();
    passed &= testRandomSmoke();
    return passed;
  }

 private:
  Scalar const kSmallEps = Constants<Scalar>::epsilon();

  Eigen::Matrix<Scalar, N - 1, 1> map(
      Eigen::Matrix<Scalar, N, N> const& T,
      Eigen::Matrix<Scalar, N - 1, 1> const& p) {
    return T.template topLeftCorner<N - 1, N - 1>() * p +
           T.template topRightCorner<N - 1, 1>();
  }

  Eigen::Matrix<Scalar, N, 1> map(Eigen::Matrix<Scalar, N, N> const& T,
                                  Eigen::Matrix<Scalar, N, 1> const& p) {
    return T * p;
  }

  std::vector<LieGroup, Eigen::aligned_allocator<LieGroup>> group_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

template <class T>
std::vector<SE3<T>, Eigen::aligned_allocator<SE3<T>>> getTestSE3s() {
  T const kPi = Constants<T>::pi();
  std::vector<SE3<T>, Eigen::aligned_allocator<SE3<T>>> se3_vec;
  se3_vec.push_back(
      SE3<T>(SO3<T>::exp(Vector3<T>(0.2, 0.5, 0.0)), Vector3<T>(0, 0, 0)));
  se3_vec.push_back(
      SE3<T>(SO3<T>::exp(Vector3<T>(0.2, 0.5, -1.0)), Vector3<T>(10, 0, 0)));
  se3_vec.push_back(SE3<T>::trans(0, 100, 5));
  se3_vec.push_back(SE3<T>::rotZ(0.00001));
  se3_vec.push_back(SE3<T>::trans(0, -0.00000001, 0.0000000001) *
                    SE3<T>::rotZ(0.00001));
  se3_vec.push_back(SE3<T>::transX(0.01) * SE3<T>::rotZ(0.00001));
  se3_vec.push_back(SE3<T>::trans(4, -5, 0) * SE3<T>::rotX(kPi));
  se3_vec.push_back(
      SE3<T>(SO3<T>::exp(Vector3<T>(0.2, 0.5, 0.0)), Vector3<T>(0, 0, 0)) *
      SE3<T>::rotX(kPi) *
      SE3<T>(SO3<T>::exp(Vector3<T>(-0.2, -0.5, -0.0)), Vector3<T>(0, 0, 0)));
  se3_vec.push_back(
      SE3<T>(SO3<T>::exp(Vector3<T>(0.3, 0.5, 0.1)), Vector3<T>(2, 0, -7)) *
      SE3<T>::rotX(kPi) *
      SE3<T>(SO3<T>::exp(Vector3<T>(-0.3, -0.5, -0.1)), Vector3<T>(0, 6, 0)));
  return se3_vec;
}

template <class T>
std::vector<SE2<T>, Eigen::aligned_allocator<SE2<T>>> getTestSE2s() {
  T const kPi = Constants<T>::pi();
  std::vector<SE2<T>, Eigen::aligned_allocator<SE2<T>>> se2_vec;
  se2_vec.push_back(SE2<T>());
  se2_vec.push_back(SE2<T>(SO2<T>(0.2), Vector2<T>(10, 0)));
  se2_vec.push_back(SE2<T>::transY(100));
  se2_vec.push_back(SE2<T>(SO2<T>(-1.), Vector2<T>(20, -1)));
  se2_vec.push_back(
      SE2<T>(SO2<T>(0.00001), Vector2<T>(-0.00000001, 0.0000000001)));
  se2_vec.push_back(SE2<T>(SO2<T>(0.3), Vector2<T>(2, 0)) * SE2<T>::rot(kPi) *
                    SE2<T>(SO2<T>(-0.3), Vector2<T>(0, 6)));
  return se2_vec;
}
}
#endif  // TESTS_HPP
