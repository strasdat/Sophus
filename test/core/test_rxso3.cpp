#include <iostream>

#include <sophus/rxso3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::RxSO3<double>>;
template class Map<Sophus::RxSO3<double> const>;
}

namespace Sophus {

template class RxSO3<double>;

template <class Scalar>
class Tests {
 public:
  using SO3Type = SO3<Scalar>;
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename RxSO3<Scalar>::Point;
  using Tangent = typename RxSO3<Scalar>::Tangent;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, 0.0, 1.)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, -1.0, 1.1)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0., 0., 0., 1.1)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0.)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0.00001)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(kPi, 0, 0, 0.9)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, 0.0, 0)) *
                         RxSO3Type::exp(Tangent(kPi, 0, 0, 0.0)) *
                         RxSO3Type::exp(Tangent(-0.2, -0.5, -0.0, 0)));
    rxso3_vec_.push_back(RxSO3Type::exp(Tangent(0.3, 0.5, 0.1, 0)) *
                         RxSO3Type::exp(Tangent(kPi, 0, 0, 0)) *
                         RxSO3Type::exp(Tangent(-0.3, -0.5, -0.1, 0)));

    Tangent tmp;
    tmp << 0, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 1, 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 1, 0, 0, 0.1;
    tangent_vec_.push_back(tmp);
    tmp << 0, 1, 0, 0.1;
    tangent_vec_.push_back(tmp);
    tmp << 0, 0, 1, -0.1;
    tangent_vec_.push_back(tmp);
    tmp << -1, 1, 0, -0.1;
    tangent_vec_.push_back(tmp);
    tmp << 20, -1, 0, 2;
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(1, 2, 4));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testSaturation();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<RxSO3Type> tests(rxso3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testSaturation() {
    bool passed = true;
    RxSO3Type small1(Constants<Scalar>::epsilon(), SO3Type());
    RxSO3Type small2(
        Constants<Scalar>::epsilon(),
        SO3Type::exp(Vector3<Scalar>(Constants<Scalar>::pi(), 0, 0)));
    RxSO3Type saturated_product = small1 * small2;
    SOPHUS_TEST_APPROX(passed, saturated_product.scale(),
                       Constants<Scalar>::epsilon(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, saturated_product.so3().matrix(),
                       (small1.so3() * small2.so3()).matrix(),
                       Constants<Scalar>::epsilon());
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw = {0, 1, 0, 0};
    Eigen::Map<RxSO3Type const> map_of_const_rxso3(raw.data());
    SOPHUS_TEST_APPROX(passed, map_of_const_rxso3.quaternion().coeffs().eval(),
                       raw, Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_const_rxso3.quaternion().coeffs().data(),
                      raw.data());
    Eigen::Map<RxSO3Type const> const_shallow_copy = map_of_const_rxso3;
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.quaternion().coeffs().eval(),
                      map_of_const_rxso3.quaternion().coeffs().eval());

    Eigen::Matrix<Scalar, 4, 1> raw2 = {1, 0, 0, 0};
    Eigen::Map<RxSO3Type> map_of_rxso3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2;
    map_of_rxso3.setQuaternion(quat);
    SOPHUS_TEST_APPROX(passed, map_of_rxso3.quaternion().coeffs().eval(), raw2,
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_rxso3.quaternion().coeffs().data(),
                      raw.data());
    SOPHUS_TEST_NEQ(passed, map_of_rxso3.quaternion().coeffs().data(),
                    quat.coeffs().data());
    Eigen::Map<RxSO3Type> shallow_copy = map_of_rxso3;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.quaternion().coeffs().eval(),
                      map_of_rxso3.quaternion().coeffs().eval());

    RxSO3Type const const_so3(quat);
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so3.data()[i], raw2.data()[i]);
    }

    RxSO3Type so3(quat);
    for (int i = 0; i < 4; ++i) {
      so3.data()[i] = raw[i];
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, so3.data()[i], raw.data()[i]);
    }
    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    RxSO3Type rxso3;
    Scalar scale(1.2);
    rxso3.setScale(scale);
    SOPHUS_TEST_APPROX(passed, scale, rxso3.scale(),
                       Constants<Scalar>::epsilon(), "setScale");
    auto so3 = rxso3_vec_[0].so3();
    rxso3.setSO3(so3);
    SOPHUS_TEST_APPROX(passed, scale, rxso3.scale(),
                       Constants<Scalar>::epsilon(), "setScale");
    SOPHUS_TEST_APPROX(passed, RxSO3Type(scale, so3).matrix(), rxso3.matrix(),
                       Constants<Scalar>::epsilon(), "RxSO3(scale, SO3)");
    SOPHUS_TEST_APPROX(passed, RxSO3Type(scale, so3.matrix()).matrix(),
                       rxso3.matrix(), Constants<Scalar>::epsilon(),
                       "RxSO3(scale, SO3)");
    Matrix3<Scalar> R = SO3<Scalar>::exp(Point(0.2, 0.5, -1.0)).matrix();
    Matrix3<Scalar> sR = R * Scalar(1.3);
    SOPHUS_TEST_APPROX(passed, RxSO3Type(sR).matrix(), sR,
                       Constants<Scalar>::epsilon(), "RxSO3(sR)");
    rxso3.setScaledRotationMatrix(sR);
    SOPHUS_TEST_APPROX(passed, sR, rxso3.matrix(), Constants<Scalar>::epsilon(),
                       "setScaleRotationMatrix");
    rxso3.setScale(scale);
    rxso3.setRotationMatrix(R);
    SOPHUS_TEST_APPROX(passed, R, rxso3.rotationMatrix(),
                       Constants<Scalar>::epsilon(), "setRotationMatrix");
    SOPHUS_TEST_APPROX(passed, scale, rxso3.scale(),
                       Constants<Scalar>::epsilon(), "setScale");

    return passed;
  }

  std::vector<RxSO3Type, Eigen::aligned_allocator<RxSO3Type>> rxso3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

template <class Scalar>
void tests() {
  using std::vector;
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename RxSO3<Scalar>::Point;
  using Tangent = typename RxSO3<Scalar>::Tangent;

  Scalar const kPi = Constants<Scalar>::pi();

  vector<RxSO3Type, Eigen::aligned_allocator<RxSO3Type>> rxso3_vec;
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, 0.0, 1.)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, -1.0, 1.1)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0., 1.1)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0.)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0.00001)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(kPi, 0, 0, 0.9)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, 0.0, 0)) *
                      RxSO3Type::exp(Tangent(kPi, 0, 0, 0.0)) *
                      RxSO3Type::exp(Tangent(-0.2, -0.5, -0.0, 0)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.3, 0.5, 0.1, 0)) *
                      RxSO3Type::exp(Tangent(kPi, 0, 0, 0)) *
                      RxSO3Type::exp(Tangent(-0.3, -0.5, -0.1, 0)));

  vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  Tangent tmp;
  tmp << 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0, 1, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0, 0, 1, -0.1;
  tangent_vec.push_back(tmp);
  tmp << -1, 1, 0, -0.1;
  tangent_vec.push_back(tmp);
  tmp << 20, -1, 0, 2;
  tangent_vec.push_back(tmp);

  vector<Point, Eigen::aligned_allocator<Point>> point_vec;
  point_vec.push_back(Point(1, 2, 4));

  LieGroupTests<RxSO3Type> tests;
  tests.setGroupElements(rxso3_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  bool passed = tests.doAllTestsPass();

  // TODO: Add proper unit tests for all functions.

  processTestResult(passed);
}

int test_rxso3() {
  using std::cerr;
  using std::endl;

  cerr << "Test RxSO3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}

}  // Sophus

int main() { return Sophus::test_rxso3(); }
