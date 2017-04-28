#include <iostream>

#include <sophus/interpolate.hpp>
#include <sophus/so3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SO3<double>>;
template class Map<Sophus::SO3<double> const>;
}

namespace Sophus {

template class SO3<double, Eigen::AutoAlign>;
template class SO3<float, Eigen::DontAlign>;

template <class Scalar>
class Tests {
 public:
  using SO3Type = SO3<Scalar>;
  using Point = typename SO3<Scalar>::Point;
  using Tangent = typename SO3<Scalar>::Tangent;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    so3_vec_.push_back(SO3Type(Eigen::Quaternion<Scalar>(0.1e-11, 0., 1., 0.)));
    so3_vec_.push_back(
        SO3Type(Eigen::Quaternion<Scalar>(-1, 0.00001, 0.0, 0.0)));
    so3_vec_.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)));
    so3_vec_.push_back(SO3Type::exp(Point(0.2, 0.5, -1.0)));
    so3_vec_.push_back(SO3Type::exp(Point(0., 0., 0.)));
    so3_vec_.push_back(SO3Type::exp(Point(0., 0., 0.00001)));
    so3_vec_.push_back(SO3Type::exp(Point(kPi, 0, 0)));
    so3_vec_.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)) *
                       SO3Type::exp(Point(kPi, 0, 0)) *
                       SO3Type::exp(Point(-0.2, -0.5, -0.0)));
    so3_vec_.push_back(SO3Type::exp(Point(0.3, 0.5, 0.1)) *
                       SO3Type::exp(Point(kPi, 0, 0)) *
                       SO3Type::exp(Point(-0.3, -0.5, -0.1)));
    tangent_vec_.push_back(Tangent(0, 0, 0));
    tangent_vec_.push_back(Tangent(1, 0, 0));
    tangent_vec_.push_back(Tangent(0, 1, 0));
    tangent_vec_.push_back(Tangent(kPi / 2., kPi / 2., 0.0));
    tangent_vec_.push_back(Tangent(-1, 1, 0));
    tangent_vec_.push_back(Tangent(20, -1, 0));
    tangent_vec_.push_back(Tangent(30, 5, -1));

    point_vec_.push_back(Point(1, 2, 4));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testUnity();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<SO3Type> tests(so3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testUnity() {
    bool passed = true;
    // Test that the complex number magnitude stays close to one.
    SO3Type current_q;
    for (std::size_t i = 0; i < 1000; ++i) {
      for (SO3Type const& q : so3_vec_) {
        current_q *= q;
      }
    }
    SOPHUS_TEST_APPROX(passed, current_q.unit_quaternion().norm(), Scalar(1),
                       Constants<Scalar>::epsilon(), "Magnitude drift");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw = {0, 1, 0, 0};
    Eigen::Map<SO3Type const> map_of_const_so3(raw.data());
    SOPHUS_TEST_APPROX(passed,
                       map_of_const_so3.unit_quaternion().coeffs().eval(), raw,
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_so3.unit_quaternion().coeffs().data(), raw.data());
    Eigen::Map<SO3Type const> const_shallow_copy = map_of_const_so3;
    SOPHUS_TEST_EQUAL(passed,
                      const_shallow_copy.unit_quaternion().coeffs().eval(),
                      map_of_const_so3.unit_quaternion().coeffs().eval());

    Eigen::Matrix<Scalar, 4, 1> raw2 = {1, 0, 0, 0};
    Eigen::Map<SO3Type> map_of_so3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2;
    map_of_so3.setQuaternion(quat);
    SOPHUS_TEST_APPROX(passed, map_of_so3.unit_quaternion().coeffs().eval(),
                       raw2, Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_so3.unit_quaternion().coeffs().data(),
                      raw.data());
    SOPHUS_TEST_NEQ(passed, map_of_so3.unit_quaternion().coeffs().data(),
                    quat.coeffs().data());
    Eigen::Map<SO3Type> shallow_copy = map_of_so3;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.unit_quaternion().coeffs().eval(),
                      map_of_so3.unit_quaternion().coeffs().eval());

    SO3Type const const_so3(quat);
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so3.data()[i], raw2.data()[i]);
    }

    SO3Type so3(quat);
    for (int i = 0; i < 4; ++i) {
      so3.data()[i] = raw[i];
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, so3.data()[i], raw.data()[i]);
    }

    SOPHUS_TEST_EQUAL(passed, SO3Type::rotX(0.2).matrix(),
                      SO3Type::exp(Point(0.2, 0, 0)).matrix());
    SOPHUS_TEST_EQUAL(passed, SO3Type::rotY(-0.2).matrix(),
                      SO3Type::exp(Point(0, -0.2, 0)).matrix());
    SOPHUS_TEST_EQUAL(passed, SO3Type::rotZ(1.1).matrix(),
                      SO3Type::exp(Point(0, 0, 1.1)).matrix());

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Matrix3<Scalar> R = so3_vec_.front().matrix();
    SO3Type so3(R);
    SOPHUS_TEST_APPROX(passed, R, so3.matrix(), Constants<Scalar>::epsilon());
    return passed;
  }

  std::vector<SO3Type, Eigen::aligned_allocator<SO3Type>> so3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int test_so3() {
  using std::cerr;
  using std::endl;

  cerr << "Test SO3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_so3(); }
