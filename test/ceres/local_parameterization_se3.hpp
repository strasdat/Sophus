#pragma once

#include <ceres/local_parameterization.h>
#include <sophus/se3.hpp>

namespace Sophus
{

class LocalParameterizationSe3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSe3() {}
  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
  {
        const Eigen::Map<const Sophus::SE3d> T(x);
        const Eigen::Map<const Eigen::Matrix<double,6,1> > dx(delta);
        Eigen::Map<Sophus::SE3d> Tdx(x_plus_delta);
        Tdx = T * Sophus::SE3d::exp(dx);
        return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const
  {

	/* Explicit formulation. Needs to be optimized */
	const double q1	   = x[0];
	const double q2	   = x[1];
	const double q3	   = x[2];
    const double q0	   = x[3];
	const double half_q0 = 0.5*q0;
    const double half_q1 = 0.5*q1;
	const double half_q2 = 0.5*q2;
	const double half_q3 = 0.5*q3;
    
    // d output_quaternion / d update
    jacobian[0] = 0.0;
	jacobian[1] = 0.0;
	jacobian[2] = 0.0;
	jacobian[3] = half_q0;
	jacobian[4] = -half_q3;
	jacobian[5] = half_q2;
    
	jacobian[6] = 0.0;
	jacobian[7] = 0.0;
	jacobian[8] = 0.0;
	jacobian[9] = half_q3;
	jacobian[10] = half_q0;
	jacobian[11] = -half_q1;
    
	jacobian[12] = 0.0;
	jacobian[13] = 0.0;
	jacobian[14] = 0.0;
	jacobian[15] = -half_q2;
	jacobian[16] = half_q1;
	jacobian[17] = half_q0;
    
	jacobian[18] = 0.0;
	jacobian[19] = 0.0;
	jacobian[20] = 0.0;
	jacobian[21] = -half_q1;
	jacobian[22] = -half_q2;
	jacobian[23] = -half_q3;    

    // d output_translation / d update
	jacobian[24]  = 1.0 - 2.0*q2*q2 - 2.0*q3*q3;
	jacobian[25]  = 2.0*q1*q2 - 2.0*q0*q3;
	jacobian[26]  = 2.0*q1*q3 + 2.0*q0*q2;
	jacobian[27]  = 0.0;
	jacobian[28]  = 0.0;
	jacobian[29]  = 0.0;
    
    jacobian[30]  = 2.0*q1*q2 + 2.0*q0*q3;
	jacobian[31]  = 1.0 - 2.0*q1*q1 - 2.0*q3*q3;
	jacobian[32]  = 2.0*q2*q3 - 2.0*q0*q1;
	jacobian[33]  = 0.0;
	jacobian[34] = 0.0;
	jacobian[35] = 0.0;
    
	jacobian[36] = 2.0*q1*q3 - 2.0*q0*q2 ;
	jacobian[37] = 2.0*q2*q3 + 2.0*q0*q1;
	jacobian[38] = 1.0 - 2.0*q1*q1 - 2.0*q2*q2;
	jacobian[39] = 0.0;
	jacobian[40] = 0.0;
	jacobian[41] = 0.0;
    
	return true;
  }

  virtual int GlobalSize() const { return 7; }
  virtual int LocalSize() const { return 6; }

};

}
