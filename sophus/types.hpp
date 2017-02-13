#ifndef SOPHUS_TYEPES_HPP
#define SOPHUS_TYEPES_HPP

#include "common.hpp"

namespace Sophus {

template <class Scalar, int M>
using Vector = Eigen::Matrix<Scalar, M, 1>;

template <class Scalar>
using Vector2 = Vector<Scalar, 2>;
using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

template <class Scalar>
using Vector3 = Vector<Scalar, 3>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <class Scalar>
using Vector4 = Vector<Scalar, 4>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

template <class Scalar>
using Vector6 = Vector<Scalar, 6>;
using Vector6f = Vector6<float>;
using Vector6d = Vector6<double>;

template <class Scalar>
using Vector7 = Vector<Scalar, 7>;
using Vector7f = Vector7<float>;
using Vector7d = Vector7<double>;

template <class Scalar, int M, int N>
using Matrix = Eigen::Matrix<Scalar, M, N>;

template <class Scalar>
using Matrix2 = Matrix<Scalar, 2, 2>;
using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;

template <class Scalar>
using Matrix3 = Matrix<Scalar, 3, 3>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <class Scalar>
using Matrix4 = Matrix<Scalar, 4, 4>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

template <class Scalar>
using Matrix6 = Matrix<Scalar, 6, 6>;
using Matrix6f = Matrix6<float>;
using Matrix6d = Matrix6<double>;

template <class Scalar>
using Matrix7 = Matrix<Scalar, 7, 7>;
using Matrix7f = Matrix7<float>;
using Matrix7d = Matrix7<double>;

}  // namespace Sophus

#endif  // SOPHUS_TYEPES_HPP
