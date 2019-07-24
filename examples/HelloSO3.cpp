#include "sophus/geometry.hpp"
#include <iostream>

int main() {

    // The following demonstrates the group multiplication of rotation matrices

    // Create rotation matrices from rotations around the x and y and z axes:
    Sophus::SO3d R1 = Sophus::SO3d::rotX(M_PI/4);
    Sophus::SO3d R2 = Sophus::SO3d::rotY(M_PI/6);
    Sophus::SO3d R3 = Sophus::SO3d::rotZ(-M_PI/3);

    std::cout << "The rotation matrices are" << std::endl;
    std::cout << "R1:\n" << R1.matrix() << std::endl;
    std::cout << "R2:\n" << R2.matrix() << std::endl;
    std::cout << "R3:\n" << R3.matrix() << std::endl;
    std::cout << "Their product R1*R2*R3:\n" << (R1*R2*R3).matrix() << std::endl;

    // Creating vectors is the same as in Eigen
    Sophus::Vector3d x;
    x << 0.0, 0.0, 1.0;

    // Rotation matrices can act on vectors
    std::cout << std::endl;
    std::cout << "Rotation matrices can act on vectors" << std::endl;
    std::cout << "x\n" << x << std::endl;
    std::cout << "R2*x\n" << R2*x << std::endl;
    std::cout << "R1*(R2*x)\n" << R1*(R2*x) << std::endl;
    std::cout << "(R1*R2)*x\n" << (R1*R2)*x << std::endl;

    // You can also use SO(3) objects as unit quaternions.
    auto q1 = R1.unit_quaternion();
    auto q2 = R2.unit_quaternion();
    auto q3 = R3.unit_quaternion();
    auto q1xq2xq3 = q1*q2*q3;
    Sophus::SO3d R4 = Sophus::SO3d(q1xq2xq3);

    std::cout << "Using unit quaternions, the product R1*R2*R3 is:\n" << R4.matrix() << std::endl;
    
}