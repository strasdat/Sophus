// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <sophus/lie/so3.h>

#include <iostream>

int main() {
  // The following demonstrates the group multiplication of rotation matrices

  // Create rotation matrices from rotations around the x and y and z axes:
  const double kPi = sophus::kPi<double>;
  sophus::SO3d R1 = sophus::SO3d::rotX(kPi / 4);
  sophus::SO3d R2 = sophus::SO3d::rotY(kPi / 6);
  sophus::SO3d R3 = sophus::SO3d::rotZ(-kPi / 3);

  std::cout << "The rotation matrices are" << std::endl;
  std::cout << "R1:\n" << R1.matrix() << std::endl;
  std::cout << "R2:\n" << R2.matrix() << std::endl;
  std::cout << "R3:\n" << R3.matrix() << std::endl;
  std::cout << "Their product R1*R2*R3:\n"
            << (R1 * R2 * R3).matrix() << std::endl;
  std::cout << std::endl;

  // Rotation matrices can act on vectors
  Eigen::Vector3d x;
  x << 0.0, 0.0, 1.0;
  std::cout << "Rotation matrices can act on vectors" << std::endl;
  std::cout << "x\n" << x << std::endl;
  std::cout << "R2*x\n" << R2 * x << std::endl;
  std::cout << "R1*(R2*x)\n" << R1 * (R2 * x) << std::endl;
  std::cout << "(R1*R2)*x\n" << (R1 * R2) * x << std::endl;
  std::cout << std::endl;

  // SO(3) are internally represented as unit quaternions.
  std::cout << "R1 in matrix form:\n" << R1.matrix() << std::endl;
  std::cout << "R1 in unit quaternion form:\n"
            << R1.unitQuaternion().coeffs() << std::endl;
  // Note that the order of coefficients of Eigen's quaternion class is
  // (imag0, imag1, imag2, real)
  std::cout << std::endl;
}
