#ifndef SOPHUS_CERES_LOCAL_PARAMETERIZATION_HPP
#define SOPHUS_CERES_LOCAL_PARAMETERIZATION_HPP

namespace Sophus {

/// Type trait used to distinguish mappable vector types from scalars
/// We use this class to distinguish Sophus::Vector<Scalar, N> from Scalar types
/// in LieGroup<T>::Tangent
///
/// Fortunately, ceres::Jet is not mappable
template <typename T>
struct is_mappable_type_t {
  template <typename U>
  using EigenTraits = Eigen::internal::traits<U>;
  // Eigen::Map<T> requires Eigen::internal::traits<T> type to be complete
  template <typename U>
  static auto map_test(U*)
      -> std::integral_constant<bool, sizeof(EigenTraits<U>) ==
                                          sizeof(EigenTraits<U>)>;
  static auto map_test(...) -> std::false_type;

  using type = decltype(map_test((T*)nullptr));
  static constexpr bool value = type::value;
};

template <typename T>
constexpr bool is_mappable_type_v = is_mappable_type_t<T>::value;

/// Helper for mapping tangent vectors (scalars) over pointers to data
template <typename T, typename E = void>
struct Mapper {
  using Scalar = T;
  using Map = Scalar&;
  using CMap = const Scalar&;

  static Map map(Scalar* ptr) { return *ptr; }
  static CMap map(const Scalar* ptr) { return *ptr; }
};

template <typename T>
struct Mapper<T, typename std::enable_if<is_mappable_type_v<T>>::type> {
  using Scalar = typename T::Scalar;
  using Map = Eigen::Map<T>;
  using CMap = Eigen::Map<const T>;

  static Map map(Scalar* ptr) { return Map(ptr); }
  static CMap map(const Scalar* ptr) { return CMap(ptr); }
};

/// Templated local parameterization for LieGroup [with implemented
/// LieGroup::Dx_this_mul_exp_x_at_0() ]
template <template <typename, int = 0> class LieGroup>
class LocalParameterization : public ceres::LocalParameterization {
 public:
  using LieGroupd = LieGroup<double>;
  using Tangent = typename LieGroupd::Tangent;
  using TangentMap = typename Sophus::Mapper<Tangent>::CMap;
  static int constexpr DoF = LieGroupd::DoF;
  static int constexpr num_parameters = LieGroupd::num_parameters;
  virtual ~LocalParameterization() {}

  /// LieGroup plus operation for Ceres
  ///
  ///  T * exp(x)
  ///
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
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
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<LieGroupd const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, num_parameters, DoF,
                             DoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const { return LieGroupd::num_parameters; }

  virtual int LocalSize() const { return LieGroupd::DoF; }
};

}  // namespace Sophus

#endif
