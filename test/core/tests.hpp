#ifndef SOPUHS_TESTS_HPP
#define SOPUHS_TESTS_HPP

#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>

#include <sophus/test_macros.hpp>

namespace Sophus {

template <class LieGroup>
class GenericTests {
 public:
  typedef typename LieGroup::Scalar Scalar;
  typedef typename LieGroup::Transformation Transformation;
  typedef typename LieGroup::Tangent Tangent;
  typedef typename LieGroup::Point Point;
  typedef typename LieGroup::Adjoint Adjoint;
  static const int N = LieGroup::N;
  static const int DoF = LieGroup::DoF;

  Scalar SMALL_EPS;

  GenericTests() : SMALL_EPS(Constants<Scalar>::epsilon()) {}

  void setGroupElements(
      const std::vector<LieGroup, Eigen::aligned_allocator<LieGroup>>&
          group_vec) {
    group_vec_ = group_vec;
  }

  void setTangentVectors(
      const std::vector<Tangent, Eigen::aligned_allocator<Tangent>>&
          tangent_vec) {
    tangent_vec_ = tangent_vec;
  }

  void setPoints(
      const std::vector<Point, Eigen::aligned_allocator<Point>>& point_vec) {
    point_vec_ = point_vec;
  }

  bool adjointTest() {
    bool passed = true;
    using std::cerr;
    using std::endl;
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
        SOPHUS_TEST_APPROX(passed, ad1, ad2, 20. * SMALL_EPS,
                           "Adjoint case %, %", i, j);
      }
    }
    return passed;
  }

  bool expLogTest() {
    using std::cerr;
    using std::endl;
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      Transformation T1 = group_vec_[i].matrix();
      Transformation T2 = LieGroup::exp(group_vec_[i].log()).matrix();
      SOPHUS_TEST_APPROX(passed, T1, T2, SMALL_EPS, "G - exp(log(G)) case: %",
                         i);
    }
    return passed;
  }

  bool expMapTest() {
    using std::cerr;
    using std::endl;
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      Tangent omega = tangent_vec_[i];
      Transformation exp_x = LieGroup::exp(omega).matrix();
      Transformation expmap_hat_x = (LieGroup::hat(omega)).exp();
      SOPHUS_TEST_APPROX(passed, exp_x, expmap_hat_x, 10. * SMALL_EPS,
                         "expmap(hat(x)) - exp(x) case: %", i);
    }
    return passed;
  }

  bool groupActionTest() {
    using std::cerr;
    using std::endl;
    bool passed = true;

    for (size_t i = 0; i < group_vec_.size(); ++i) {
      for (size_t j = 0; j < point_vec_.size(); ++j) {
        const Point& p = point_vec_[j];
        Transformation T = group_vec_[i].matrix();
        Point point1 = group_vec_[i] * p;
        Point point2 = map(T, p);
        SOPHUS_TEST_APPROX(passed, point1, point2, SMALL_EPS,
                           "Transform point case: %", i);
      }
    }
    return passed;
  }

  bool lieBracketTest() {
    using std::cerr;
    using std::endl;
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      for (size_t j = 0; j < tangent_vec_.size(); ++j) {
        Tangent tangent1 =
            LieGroup::lieBracket(tangent_vec_[i], tangent_vec_[j]);
        Transformation hati = LieGroup::hat(tangent_vec_[i]);
        Transformation hatj = LieGroup::hat(tangent_vec_[j]);

        Tangent tangent2 = LieGroup::vee(hati * hatj - hatj * hati);
        SOPHUS_TEST_APPROX(passed, tangent1, tangent2, SMALL_EPS,
                           "Lie Bracket case: %", i);
      }
    }
    return passed;
  }

  bool veeHatTest() {
    using std::cerr;
    using std::endl;
    bool passed = true;
    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      SOPHUS_TEST_APPROX(passed, tangent_vec_[i],
                         LieGroup::vee(LieGroup::hat(tangent_vec_[i])),
                         SMALL_EPS, "Hat-vee case: %", i);
    }
    return passed;
  }

  bool doAllTestsPass() {
    bool passed = adjointTest();
    passed &= expLogTest();
    passed &= expMapTest();
    passed &= groupActionTest();
    passed &= lieBracketTest();
    passed &= veeHatTest();
    return passed;
  }

 private:
  Eigen::Matrix<Scalar, N - 1, 1> map(
      const Eigen::Matrix<Scalar, N, N>& T,
      const Eigen::Matrix<Scalar, N - 1, 1>& p) {
    return T.template topLeftCorner<N - 1, N - 1>() * p +
           T.template topRightCorner<N - 1, 1>();
  }

  Eigen::Matrix<Scalar, N, 1> map(const Eigen::Matrix<Scalar, N, N>& T,
                                  const Eigen::Matrix<Scalar, N, 1>& p) {
    return T * p;
  }

  std::vector<LieGroup, Eigen::aligned_allocator<LieGroup>> group_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};
}
#endif  // TESTS_HPP
