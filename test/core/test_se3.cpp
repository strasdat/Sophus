#include <iostream>

#include <sophus/se3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SE3<double>>;
template class Map<Sophus::SE3<double> const>;
}

namespace Sophus {

template class SE3<double>;

template <class Scalar>
class Tests {
 public:
  using SE3Type = SE3<Scalar>;
  using SO3Type = SO3<Scalar>;
  using Point = typename SE3<Scalar>::Point;
  using Tangent = typename SE3<Scalar>::Tangent;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0.2, 0.5, -1.0)), Point(10, 0, 0)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0., 0., 0.)), Point(0, 100, 5)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0., 0., 0.00001)), Point(0, 0, 0)));
    se3_vec_.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                               Point(0, -0.00000001, 0.0000000001)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0., 0., 0.00001)), Point(0.01, 0, 0)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(4, -5, 0)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
        SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
        SE3Type(SO3Type::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)));
    se3_vec_.push_back(
        SE3Type(SO3Type::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
        SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
        SE3Type(SO3Type::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0)));

    Tangent tmp;
    tmp << 0, 0, 0, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 1, 0, 0, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 0, 1, 0, 1, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 0, -5, 10, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << -1, 1, 0, 0, 0, 1;
    tangent_vec_.push_back(tmp);
    tmp << 20, -1, 0, -1, 1, 0;
    tangent_vec_.push_back(tmp);
    tmp << 30, 5, -1, 20, -1, 0;
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(1, 2, 4));
    point_vec_.push_back(Point(1, -3, 0.5));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<SE3Type> tests(se3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 7, 1> raw;
    raw << 0, 1, 0, 0, 1, 3, 2;
    Eigen::Map<SE3Type const> map_of_const_se3(raw.data());
    SOPHUS_TEST_APPROX(
        passed, map_of_const_se3.unit_quaternion().coeffs().eval(),
        raw.template head<4>().eval(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, map_of_const_se3.translation().eval(),
                       raw.template tail<3>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_se3.unit_quaternion().coeffs().data(), raw.data());
    SOPHUS_TEST_EQUAL(passed, map_of_const_se3.translation().data(),
                      raw.data() + 4);
    Eigen::Map<SE3Type const> const_shallow_copy = map_of_const_se3;
    SOPHUS_TEST_EQUAL(passed,
                      const_shallow_copy.unit_quaternion().coeffs().eval(),
                      map_of_const_se3.unit_quaternion().coeffs().eval());
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.translation().eval(),
                      map_of_const_se3.translation().eval());

    Eigen::Matrix<Scalar, 7, 1> raw2;
    raw2 << 1, 0, 0, 0, 3, 2, 1;
    Eigen::Map<SE3Type> map_of_se3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2.template head<4>();
    map_of_se3.setQuaternion(quat);
    map_of_se3.translation() = raw2.template tail<3>();
    SOPHUS_TEST_APPROX(passed, map_of_se3.unit_quaternion().coeffs().eval(),
                       raw2.template head<4>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, map_of_se3.translation().eval(),
                       raw2.template tail<3>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_se3.unit_quaternion().coeffs().data(),
                      raw.data());
    SOPHUS_TEST_EQUAL(passed, map_of_se3.translation().data(), raw.data() + 4);
    SOPHUS_TEST_NEQ(passed, map_of_se3.unit_quaternion().coeffs().data(),
                    quat.coeffs().data());
    Eigen::Map<SE3Type> shallow_copy = map_of_se3;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.unit_quaternion().coeffs().eval(),
                      map_of_se3.unit_quaternion().coeffs().eval());
    SOPHUS_TEST_EQUAL(passed, shallow_copy.translation().eval(),
                      map_of_se3.translation().eval());
    Eigen::Map<SE3Type> const const_map_of_se3 = map_of_se3;
    SOPHUS_TEST_EQUAL(passed,
                      const_map_of_se3.unit_quaternion().coeffs().eval(),
                      map_of_se3.unit_quaternion().coeffs().eval());
    SOPHUS_TEST_EQUAL(passed, const_map_of_se3.translation().eval(),
                      map_of_se3.translation().eval());

    SE3Type const const_se3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_se3.data()[i], raw2.data()[i]);
    }

    SE3Type se3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw2.data()[i]);
    }

    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw.data()[i]);
    }
    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 4> I = Eigen::Matrix<Scalar, 4, 4>::Identity();
    SOPHUS_TEST_EQUAL(passed, SE3Type().matrix(), I);

    SE3Type se3 = se3_vec_.front();
    Point translation = se3.translation();
    SO3Type so3 = se3.so3();

    SOPHUS_TEST_APPROX(passed, SE3Type(so3, translation).matrix(), se3.matrix(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, SE3Type(so3.matrix(), translation).matrix(),
                       se3.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed,
                       SE3Type(so3.unit_quaternion(), translation).matrix(),
                       se3.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, SE3Type(se3.matrix()).matrix(), se3.matrix(),
                       Constants<Scalar>::epsilon());
    return passed;
  }

  std::vector<SE3Type, Eigen::aligned_allocator<SE3Type>> se3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int test_se3() {
  using std::cerr;
  using std::endl;

  cerr << "Test SE3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_se3(); }
