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

namespace details {
template <class Scalar>
class Metric {
 public:
  static Scalar impl(Scalar s0, Scalar s1) {
    using std::abs;
    return abs(s0 - s1);
  }
};

template <class Scalar, int M, int N>
class Metric<Matrix<Scalar, M, N>> {
 public:
  static Scalar impl(Matrix<Scalar, M, N> const& p0,
                     Matrix<Scalar, M, N> const& p1) {
    return (p0 - p1).norm();
  }
};

template <typename Scalar>
class SetToZero {
 public:
  static void impl(Scalar& s) { s = Scalar(0); }
};

template <typename Scalar, int M, int N>
class SetToZero<Matrix<Scalar, M, N>> {
 public:
  static void impl(Matrix<Scalar, M, N>& v) { v.setZero(); }
};

template <typename Scalar>
class SquaredNorm {
 public:
  static Scalar impl(Scalar const& s) { return s * s; }
};

template <typename Scalar, int N>
class SquaredNorm<Matrix<Scalar, N, 1>> {
 public:
  static Scalar impl(Matrix<Scalar, N, 1> const& s) { return s.squaredNorm(); }
};

template <typename Scalar>
class Transpose {
 public:
  static Scalar impl(Scalar const& s) { return s; }
};

template <typename Scalar, int M, int N>
class Transpose<Matrix<Scalar, M, N>> {
 public:
  static Matrix<Scalar, M, N> impl(Matrix<Scalar, M, N> const& s) {
    return s.transpose();
  }
};
}  // namespace details

// Returns Euclidiean metric between two points ``p0`` and ``p1``, with ``p``
// being a matrix or a scalar.
//
template <class T>
auto metric(T const& p0, T const& p1)
    -> decltype(details::Metric<T>::impl(p0, p1)) {
  return details::Metric<T>::impl(p0, p1);
}

// Sets point ``p`` to zero, with ``p`` being a matrix or a scalar.
//
template <class T>
auto setToZero(T& p) -> decltype(details::SetToZero<T>::impl(p)) {
  return details::SetToZero<T>::impl(p);
}

// Returns the squared 2-norm of ``p``, with ``p`` being a vector or a scalar.
//
template <class T>
auto squaredNorm(T const& p) -> decltype(details::SquaredNorm<T>::impl(p)) {
  return details::SquaredNorm<T>::impl(p);
}

// Returns ``p.transpose()`` if ``p`` is a matrix, and simply ``p`` if m is a
// scalar.
//
template <typename T>
auto transpose(T const& p) -> decltype(details::Transpose<T>::impl(T())) {
  return details::Transpose<T>::impl(p);
}

}  // namespace Sophus

#endif  // SOPHUS_TYEPES_HPP
