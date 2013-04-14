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

inline double tan  (double x) { return std::tan(x);      }
inline double atan (double x) { return std::atan(x);     }

template <typename T, int N> inline
ceres::Jet<T, N> fabs(const ceres::Jet<T, N>& f) {
    return abs(f);
}

// tan(a+h) ~= tan(a) + 1 / cos(a)^2 h
template <typename T, int N> inline
ceres::Jet<T, N> tan(const ceres::Jet<T, N>& f) {
    const T cosf = cos(f.a);
    Jet<T, N> g;
    g.a = tan(f.a);
    g.v = f.v / (cosf*cosf);    
    return g;
}

// atan(a) ~= atan(a) + 1 / (x^2+1) h
template <typename T, int N> inline
ceres::Jet<T, N> atan(const Jet<T, N>& f) {
    Jet<T, N> g;
    g.a = atan(f.a);
    g.v = f.v / (f.a*f.a + 1);
    return g;
}

}
