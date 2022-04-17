#pragma once

#include <ceres/local_parameterization.h>

namespace Sophus {

template <class T, std::size_t = sizeof(T)>
constexpr std::true_type complete(T*);
constexpr std::false_type complete(...);

template <class T>
using IsSpecialized = decltype(complete(std::declval<T*>()));

/// Type trait used to distinguish mappable vector types from scalars
///
/// We use this class to distinguish Sophus::Vector<Scalar, N> from Scalar types
/// in LieGroup<T>::Tangent
///
/// Primary use is mapping LieGroup::Tangent over raw data, with 2 options:
///  - LieGroup::Tangent is "scalar" (for SO2), then we just dereference pointer
///  - LieGroup::Tangent is Sophus::Vector<...>, then we need to use Eigen::Map
///
/// Specialization of Eigen::internal::traits<T> for T is crucial for
/// for constructing Eigen::Map<T>, thus we use that property for distinguishing
/// between those two options.
/// At this moment there seem to be no option to check this using only
/// "external" API of Eigen
template <class T>
using IsMappable = IsSpecialized<Eigen::internal::traits<std::decay_t<T>>>;

template <class T>
constexpr bool IsMappableV = IsMappable<T>::value;

/// Helper for mapping tangent vectors (scalars) over pointers to data
template <typename T, typename E = void>
struct Mapper {
  using Scalar = T;
  using Map = Scalar&;
  using ConstMap = const Scalar&;

  static Map map(Scalar* ptr) noexcept { return *ptr; }
  static ConstMap map(const Scalar* ptr) noexcept { return *ptr; }
};

template <typename T>
struct Mapper<T, typename std::enable_if<IsMappableV<T>>::type> {
  using Scalar = typename T::Scalar;
  using Map = Eigen::Map<T>;
  using ConstMap = Eigen::Map<const T>;

  static Map map(Scalar* ptr) noexcept { return Map(ptr); }
  static ConstMap map(const Scalar* ptr) noexcept { return ConstMap(ptr); }
};

/// Templated local parameterization for LieGroup [with implemented
/// LieGroup::Dx_this_mul_exp_x_at_0() ]
template <template <typename, int = 0> class LieGroup>
class LocalParameterization : public ceres::LocalParameterization {
 public:
  using LieGroupd = LieGroup<double>;
  using Tangent = typename LieGroupd::Tangent;
  using TangentMap = typename Sophus::Mapper<Tangent>::ConstMap;
  static int constexpr DoF = LieGroupd::DoF;
  static int constexpr num_parameters = LieGroupd::num_parameters;

  /// LieGroup plus operation for Ceres
  ///
  ///  T * exp(x)
  ///
  bool Plus(double const* T_raw, double const* delta_raw,
            double* T_plus_delta_raw) const override {
    Eigen::Map<LieGroupd const> const T(T_raw);
    TangentMap delta = Sophus::Mapper<Tangent>::map(delta_raw);
    Eigen::Map<LieGroupd> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * LieGroupd::exp(delta);
    return true;
  }

  /// Jacobian of LieGroup plus operation for Ceres
  ///
  /// Dx T * exp(x)  with  x=0
  ///
  bool ComputeJacobian(double const* T_raw,
                       double* jacobian_raw) const override {
    Eigen::Map<LieGroupd const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, num_parameters, DoF,
                             DoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  int GlobalSize() const override { return LieGroupd::num_parameters; }

  int LocalSize() const override { return LieGroupd::DoF; }
};

}  // namespace Sophus
