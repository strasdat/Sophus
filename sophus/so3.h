// This file is part of Sophus.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef SOPHUS_SO3_H
#define SOPHUS_SO3_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>


namespace Sophus
{
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

template<typename Scalar> class SO3Group;
typedef SO3Group<double> SO3;

////////////////////////////////////////////////////////////////////////////
// Template Definition
////////////////////////////////////////////////////////////////////////////

const double SMALL_EPS = 1e-10;

template<typename Scalar=double>
class SO3Group
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SO3Group                  ();

  SO3Group                  (const SO3Group & other);

  explicit
  SO3Group                  (const Matrix<Scalar,3,3> & _R);

  explicit
  SO3Group                  (const Quaternion<Scalar> & unit_quaternion);

  SO3Group                  (Scalar rot_x,
                              Scalar rot_y,
                              Scalar rot_z);
  void
  operator=                  (const SO3Group<Scalar> & SO3Group);

  SO3Group
  operator*                  (const SO3Group<Scalar> & SO3Group) const;

  void
  operator*=                 (const SO3Group<Scalar> & SO3Group);

  Matrix<Scalar,3,1>
  operator*                  (const Matrix<Scalar,3,1> & xyz) const;

  SO3Group
  inverse                    () const;

  Matrix<Scalar,3,3>
  matrix                     () const;

  Matrix<Scalar,3,3>
  Adj                        () const;

  Matrix<Scalar,3,3>
  generator                  (int i);

  Matrix<Scalar,3,1>
  log                        () const;

  static SO3Group
  exp                        (const Matrix<Scalar,3,1> & omega);

  static SO3Group
  expAndTheta                (const Matrix<Scalar,3,1> & omega,
                              Scalar * theta);
  static Matrix<Scalar,3,1>
  log                        (const SO3Group & SO3Group);

  static Matrix<Scalar,3,1>
  logAndTheta                (const SO3Group & SO3Group,
                              Scalar * theta);

  static Matrix<Scalar,3,3>
  hat                        (const Matrix<Scalar,3,1> & omega);

  static Matrix<Scalar,3,1>
  vee                        (const Matrix<Scalar,3,3> & Omega);

  static Matrix<Scalar,3,1>
  lieBracket                 (const Matrix<Scalar,3,1> & omega1,
                              const Matrix<Scalar,3,1> & omega2);

  static Matrix<Scalar,3,3>
  d_lieBracketab_by_d_a      (const Matrix<Scalar,3,1> & b);

  void
  setQuaternion              (const Quaternion<Scalar>& quaternion);

  const Quaternion<Scalar> & unit_quaternion() const
  {
    return unit_quaternion_;
  }

  Quaternion<Scalar> & unit_quaternion()
  {
    return unit_quaternion_;
  }

  static const int DoF = 3;

protected:
  Quaternion<Scalar> unit_quaternion_;
};

////////////////////////////////////////////////////////////////////////////
// Template Implementation
////////////////////////////////////////////////////////////////////////////

template<typename Scalar> inline
SO3Group<Scalar>::SO3Group()
{
  unit_quaternion_.setIdentity();
}

template<typename Scalar> inline
SO3Group<Scalar>
::SO3Group(const SO3Group & other) : unit_quaternion_(other.unit_quaternion_) {}

template<typename Scalar>
SO3Group<Scalar>
::SO3Group(const Matrix<Scalar,3,3> & R) : unit_quaternion_(R) {}

template<typename Scalar> inline
SO3Group<Scalar>
::SO3Group(const Quaternion<Scalar> & quat) : unit_quaternion_(quat)
{
  assert(unit_quaternion_.squaredNorm() > SMALL_EPS);
  unit_quaternion_.normalize();
}

template<typename Scalar> inline
SO3Group<Scalar>
::SO3Group(Scalar rot_x, Scalar rot_y, Scalar rot_z)
{
  unit_quaternion_
      = (SO3Group::exp(Matrix<Scalar,3,1>(rot_x, 0.f, 0.f))
         *SO3Group::exp(Matrix<Scalar,3,1>(0.f, rot_y, 0.f))
         *SO3Group::exp(Matrix<Scalar,3,1>(0.f, 0.f, rot_z))).unit_quaternion_;
}

template<typename Scalar> inline
void SO3Group<Scalar>
::operator=(const SO3Group<Scalar> & other)
{
  this->unit_quaternion_ = other.unit_quaternion_;
}

template<typename Scalar> inline
SO3Group<Scalar> SO3Group<Scalar>
::operator*(const SO3Group<Scalar>& other) const
{
  SO3Group<Scalar> result(*this);
  result.unit_quaternion_ *= other.unit_quaternion_;
  result.unit_quaternion_.normalize();
  return result;
}

template<typename Scalar> inline
void SO3Group<Scalar>
::operator*=(const SO3Group<Scalar>& other)
{
  unit_quaternion_ *= other.unit_quaternion_;
  unit_quaternion_.normalize();
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SO3Group<Scalar>
::operator*(const Matrix<Scalar,3,1> & xyz) const
{
  return unit_quaternion_._transformVector(xyz);
}

template<typename Scalar> inline
SO3Group<Scalar> SO3Group<Scalar>
::inverse() const
{
  return SO3Group<Scalar>(unit_quaternion_.conjugate());
}

template<typename Scalar> inline
Matrix<Scalar,3,3> SO3Group<Scalar>
::matrix() const
{
  return unit_quaternion_.toRotationMatrix();
}

template<typename Scalar> inline
Matrix<Scalar,3,3> SO3Group<Scalar>
::Adj() const
{
  return matrix();
}

template<typename Scalar> inline
Matrix<Scalar,3,3> SO3Group<Scalar>
::generator(int i)
{
  assert(i>=0 && i<3);
  Matrix<Scalar,3,1> e;
  e.setZero();
  e[i] = 1.f;
  return hat(e);
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SO3Group<Scalar>
::log() const
{
  return SO3Group<Scalar>::log(*this);
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SO3Group<Scalar>
::log(const SO3Group<Scalar> & other)
{
  Scalar theta;
  return logAndTheta(other, &theta);
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SO3Group<Scalar>
::logAndTheta(const SO3Group<Scalar> & other, Scalar * theta)
{

    Scalar n = other.unit_quaternion_.vec().norm();
    Scalar w = other.unit_quaternion_.w();
    Scalar squared_w = w*w;

    Scalar two_atan_nbyw_by_n;
    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    if (n < SMALL_EPS)
    {
      // If quaternion is normalized and n=1, then w should be 1;
      // w=0 should never happen here!
      assert(fabs(w)>SMALL_EPS);

      two_atan_nbyw_by_n = 2./w - 2.*(n*n)/(w*squared_w);
    }
    else
    {
      if (fabs(w)<SMALL_EPS)
      {
        if (w>0)
        {
          two_atan_nbyw_by_n = M_PI/n;
        }
        else
        {
          two_atan_nbyw_by_n = -M_PI/n;
        }
      }
      two_atan_nbyw_by_n = 2*atan(n/w)/n;
    }

    *theta = two_atan_nbyw_by_n*n;
    return two_atan_nbyw_by_n * other.unit_quaternion_.vec();
}

template<typename Scalar> inline
SO3Group<Scalar> SO3Group<Scalar>
::exp(const Matrix<Scalar,3,1> & omega)
{
  Scalar theta;
  return expAndTheta(omega, &theta);
}

template<typename Scalar> inline
SO3Group<Scalar> SO3Group<Scalar>
::expAndTheta(const Matrix<Scalar,3,1> & omega, Scalar * theta)
{
  *theta = omega.norm();
  Scalar half_theta = 0.5*(*theta);

  Scalar imag_factor;
  Scalar real_factor = cos(half_theta);
  if((*theta)<SMALL_EPS)
  {
    Scalar theta_sq = (*theta)*(*theta);
    Scalar theta_po4 = theta_sq*theta_sq;
    imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
  }
  else
  {
    Scalar sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta/(*theta);
  }

  return SO3Group<Scalar>(Quaternion<Scalar>(real_factor,
                         imag_factor*omega.x(),
                         imag_factor*omega.y(),
                         imag_factor*omega.z()));
}

template<typename Scalar> inline
Matrix<Scalar,3,3> SO3Group<Scalar>
::hat(const Matrix<Scalar,3,1> & v)
{
  Matrix<Scalar,3,3> Omega;
  Omega <<  0, -v(2),  v(1)
      ,  v(2),     0, -v(0)
      , -v(1),  v(0),     0;
  return Omega;
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SO3Group<Scalar>
::vee(const Matrix<Scalar,3,3> & Omega)
{
  assert(fabs(Omega(2,1)+Omega(1,2))<SMALL_EPS);
  assert(fabs(Omega(0,2)+Omega(2,0))<SMALL_EPS);
  assert(fabs(Omega(1,0)+Omega(0,1))<SMALL_EPS);
  return Matrix<Scalar,3,1>(Omega(2,1), Omega(0,2), Omega(1,0));
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SO3Group<Scalar>
::lieBracket(const Matrix<Scalar,3,1> & omega1, const Matrix<Scalar,3,1> & omega2)
{
  return omega1.cross(omega2);
}

template<typename Scalar> inline
Matrix<Scalar,3,3> SO3Group<Scalar>
::d_lieBracketab_by_d_a(const Matrix<Scalar,3,1> & b)
{
  return -hat(b);
}

template<typename Scalar> inline
void SO3Group<Scalar>::
setQuaternion(const Quaternion<Scalar>& quaternion)
{
  assert(quaternion.norm()!=0);
  unit_quaternion_ = quaternion;
  unit_quaternion_.normalize();
}

} // end namespace


#endif
