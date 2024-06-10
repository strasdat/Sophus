#ifndef SOPHUS_CERES_TYPETRAITS_HPP
#define SOPHUS_CERES_TYPETRAITS_HPP

#include "common.hpp"

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
struct Mapper<T, typename std::enable_if_t<IsMappableV<T>>> {
  using Scalar = typename T::Scalar;
  using Map = Eigen::Map<T>;
  using ConstMap = Eigen::Map<const T>;

  static Map map(Scalar* ptr) noexcept { return Map(ptr); }
  static ConstMap map(const Scalar* ptr) noexcept { return ConstMap(ptr); }
};

}  // namespace Sophus

#endif
