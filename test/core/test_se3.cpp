#include <iostream>

#include <sophus/se3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SE3<double>>;
template class Map<Sophus::SE3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class SE3<double, Eigen::AutoAlign>;
template class SE3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class SE3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
class Tests {
 public:
  using SE3Type = SE3<Scalar>;
  using SO3Type = SO3<Scalar>;
  using Point = typename SE3<Scalar>::Point;
  using Tangent = typename SE3<Scalar>::Tangent;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    se3_vec_ = getTestSE3s<Scalar>();

    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(-5), Scalar(10), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(-1), Scalar(1), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(-1), Scalar(20), Scalar(-1), Scalar(0);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testMutatingAccessors();
    passed &= testConstructors();
    passed &= testFit();
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
    raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
        Scalar(2);
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
    raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
        Scalar(1);
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
    SE3Type trans = SE3Type::transX(Scalar(0.2));
    SOPHUS_TEST_APPROX(passed, trans.translation().x(), Scalar(0.2),
                       Constants<Scalar>::epsilon());
    trans = SE3Type::transY(Scalar(0.7));
    SOPHUS_TEST_APPROX(passed, trans.translation().y(), Scalar(0.7),
                       Constants<Scalar>::epsilon());
    trans = SE3Type::transZ(Scalar(-0.2));
    SOPHUS_TEST_APPROX(passed, trans.translation().z(), Scalar(-0.2),
                       Constants<Scalar>::epsilon());
    Tangent t;
    t << Scalar(0), Scalar(0), Scalar(0), Scalar(0.2), Scalar(0), Scalar(0);
    SOPHUS_TEST_EQUAL(passed, SE3Type::rotX(Scalar(0.2)).matrix(),
                      SE3Type::exp(t).matrix());
    t << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(-0.2), Scalar(0);
    SOPHUS_TEST_EQUAL(passed, SE3Type::rotY(Scalar(-0.2)).matrix(),
                      SE3Type::exp(t).matrix());
    t << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(1.1);
    SOPHUS_TEST_EQUAL(passed, SE3Type::rotZ(Scalar(1.1)).matrix(),
                      SE3Type::exp(t).matrix());

    auto is_set = map_of_se3.trySetRotationFromQuaternion(quat);
    SOPHUS_TEST(passed, is_set);
    SOPHUS_TEST_APPROX(passed, map_of_se3.unit_quaternion().coeffs().eval(),
                       quat.coeffs().eval(), Constants<Scalar>::epsilon());

    is_set = map_of_se3.trySetRotationFromQuaternion(
        Eigen::Quaternion<Scalar>(Scalar(0), Scalar(0), Scalar(0), Scalar(0)));
    SOPHUS_TEST(passed, !is_set);

    Matrix3<Scalar> R = se3_vec_.front().rotationMatrix();
    auto is_set2 = map_of_se3.trySetRotationFromMatrix(R);
    SOPHUS_TEST(passed, is_set2);
    SOPHUS_TEST_APPROX(passed, map_of_se3.rotationMatrix(), R,
                       Constants<Scalar>::epsilon());
    Vector3<Scalar> tmp = R.col(0);
    R.col(0) = R.col(1);
    R.col(1) = tmp;
    is_set2 = map_of_se3.trySetRotationFromMatrix(R);
    SOPHUS_TEST(passed, !is_set2);
    SOPHUS_TEST(passed,
                is_set2.error() ==
                    RotationMatrixError::kOrthogonalButNegativeDeterminant);

    R(1, 1) = Scalar(2);
    is_set2 = map_of_se3.trySetRotationFromMatrix(R);
    SOPHUS_TEST(passed, !is_set2);
    SOPHUS_TEST(passed, is_set2.error() == RotationMatrixError::kNotOrthogonal);

    return passed;
  }

  bool testMutatingAccessors() {
    bool passed = true;
    SE3Type se3;
    SO3Type R(SO3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))));
    se3.setRotationMatrix(R.matrix());
    SOPHUS_TEST_APPROX(passed, se3.rotationMatrix(), R.matrix(),
                       Constants<Scalar>::epsilon());

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

    Matrix3<Scalar> R = se3.so3().matrix();
    auto se3_from_mat = SE3Type::tryFromMatrixAndTranslation(R, translation);
    SOPHUS_TEST(passed, se3_from_mat);
    SOPHUS_TEST_APPROX(passed, se3_from_mat->rotationMatrix(), R,
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, se3_from_mat->translation(), translation,
                       Constants<Scalar>::epsilon());
    R(0, 0) = Scalar(0);
    se3_from_mat = SE3Type::tryFromMatrixAndTranslation(R, translation);
    SOPHUS_TEST(passed, !se3_from_mat);

    Eigen::Quaternion<Scalar> quat = se3.so3().unit_quaternion();
    auto se3_from_quat =
        SE3Type::tryFromQuaternionAndTranslation(quat, translation);
    SOPHUS_TEST(passed, se3_from_quat);
    SOPHUS_TEST_APPROX(passed, se3_from_quat->unit_quaternion().coeffs(),
                       quat.coeffs(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, se3_from_quat->translation(), translation,
                       Constants<Scalar>::epsilon());
    se3_from_quat = SE3Type::tryFromQuaternionAndTranslation(
        Eigen::Quaternion<Scalar>(Scalar(0), Scalar(0), Scalar(0), Scalar(0)),
        translation);
    SOPHUS_TEST(passed, !se3_from_quat);

    return passed;
  }

  template <class S = Scalar>
  enable_if_t<std::is_floating_point<S>::value, bool> testFit() {
    bool passed = true;

    for (int i = 0; i < 100; ++i) {
      Matrix4<Scalar> T = Matrix4<Scalar>::Random();
      SE3Type se3 = SE3Type::fitToSE3(T);
      SE3Type se3_2 = SE3Type::fitToSE3(se3.matrix());

      SOPHUS_TEST_APPROX(passed, se3.matrix(), se3_2.matrix(),
                         Constants<Scalar>::epsilon());
    }
    for (Scalar const angle :
         {Scalar(0.0), Scalar(0.1), Scalar(0.3), Scalar(-0.7)}) {
      SOPHUS_TEST_APPROX(passed, SE3Type::rotX(angle).angleX(), angle,
                         Constants<Scalar>::epsilon());
      SOPHUS_TEST_APPROX(passed, SE3Type::rotY(angle).angleY(), angle,
                         Constants<Scalar>::epsilon());
      SOPHUS_TEST_APPROX(passed, SE3Type::rotZ(angle).angleZ(), angle,
                         Constants<Scalar>::epsilon());
    }
    return passed;
  }

  template <class S = Scalar>
  enable_if_t<!std::is_floating_point<S>::value, bool> testFit() {
    return true;
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

#if SOPHUS_CERES
  cerr << "ceres::Jet<double, 3> tests: " << endl;
  Tests<ceres::Jet<double, 3>>().runAll();
#endif

  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_se3(); }
