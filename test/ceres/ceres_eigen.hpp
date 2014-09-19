#pragma once

#include <ceres/jet.h>
#include <Eigen/Core>

namespace Eigen {

// permits to get the epsilon, dummy_precision, lowest, highest functions
template<> struct NumTraits<ceres::Jet<double,6> >
    : NumTraits<double> 
{
    typedef ceres::Jet<double,6> Real;
    typedef ceres::Jet<double,6> NonInteger;
    typedef ceres::Jet<double,6> Nested;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

}

namespace ceres {

template <typename T, int N> inline
ceres::Jet<T, N> fabs(const ceres::Jet<T, N>& f) {
    return abs(f);
}

}
