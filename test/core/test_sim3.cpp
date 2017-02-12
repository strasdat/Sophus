#include <iostream>

#include <unsupported/Eigen/MatrixFunctions>

#include <sophus/sim3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Sim3<double>>;
template class Map<Sophus::Sim3<double> const>;
}

namespace Sophus {

template class Sim3<double>;

template <class Scalar>
void tests() {
  using std::vector;
  using Sim3Type = Sim3<Scalar>;
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename Sim3<Scalar>::Point;
  using Tangent = typename Sim3<Scalar>::Tangent;
  using Vector4Type = Eigen::Matrix<Scalar, 4, 1>;
  Scalar const PI = Constants<Scalar>::pi();

  vector<Sim3Type, Eigen::aligned_allocator<Sim3Type>> sim3_vec;
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0, 1.)), Point(0, 0, 0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, -1.0, 1.1)),
                              Point(10, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0., 0.)), Point(0, 10, 5)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0., 1.1)), Point(0, 10, 5)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.)),
                              Point(0, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.0000001)),
               Point(1, -1.00000001, 2.0000000001)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0)),
                              Point(0.01, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0, 0.9)), Point(4, -5, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0, 0)), Point(0, 0, 0)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0, 0)), Point(0, 0, 0)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(-0.2, -0.5, -0.0, 0)),
               Point(0, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0.3, 0.5, 0.1, 0)), Point(2, 0, -7)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0, 0)), Point(0, 0, 0)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(-0.3, -0.5, -0.1, 0)),
               Point(0, 6, 0)));
  vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  Tangent tmp;
  tmp << 0, 0, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, 1, 0, 1, 0, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0, 0, 1, 0, 1, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << -1, 1, 0, 0, 0, 1, -0.1;
  tangent_vec.push_back(tmp);
  tmp << 20, -1, 0, -1, 1, 0, -0.1;
  tangent_vec.push_back(tmp);
  tmp << 30, 5, -1, 20, -1, 0, 1.5;
  tangent_vec.push_back(tmp);

  vector<Point, Eigen::aligned_allocator<Point>> point_vec;
  point_vec.push_back(Point(1, 2, 4));

  LieGroupTests<Sim3Type> tests;
  tests.setGroupElements(sim3_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  bool passed = tests.doAllTestsPass();

  Sim3Type sim3;
  Scalar scale(1.2);
  sim3.setScale(scale);
  SOPHUS_TEST_APPROX(passed, scale, sim3.scale(), Constants<Scalar>::epsilon(),
                     "setScale");

  sim3.setQuaternion(sim3_vec[0].rxso3().quaternion());
  SOPHUS_TEST_APPROX(passed, sim3_vec[0].rxso3().quaternion().coeffs(),
                     sim3_vec[0].rxso3().quaternion().coeffs(),
                     Constants<Scalar>::epsilon(), "setQuaternion");

  processTestResult(passed);
}

int test_sim3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Sim3" << endl << endl;
  cerr << "Double tests: " << endl;
  tests<double>();
  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_sim3(); }
