// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/se3.h"

#include <Eigen/Dense>
#include <farm_ng/core/enum/enum.h>
#include <farm_ng/core/logging/logger.h>
#include <farm_ng/core/misc/void.h>

#include <deque>
#include <map>
#include <set>
#include <variant>

namespace sophus::experimental {

FARM_ENUM(ArgType, (variable, conditioned));

template <int kTangentDim, class TManifold = Eigen::Vector<double, kTangentDim>>
struct ManifoldFamily {
  using Manifold = TManifold;

  std::vector<Manifold> manifolds;

  Manifold oplus(
      Manifold const& g, Eigen::Vector<double, kTangentDim> const& vec_a);
};

template <int kTangentDim, class TManifold>
struct Var {
  static ArgType constexpr kArgType = ArgType::variable;

  Var(ManifoldFamily<kTangentDim, TManifold> const& family) : family(family) {}
  ManifoldFamily<kTangentDim, TManifold> const& family;
};

template <int kTangentDim, class TManifold>
struct CondVar {
  static ArgType constexpr kArgType = ArgType::conditioned;

  CondVar(ManifoldFamily<kTangentDim, TManifold> const& family)
      : family(family) {}
  ManifoldFamily<kTangentDim, TManifold> const& family;
};

template <int kBlockDim>
struct LeastSquaresCostTermState {
  LeastSquaresCostTermState() {
    hessian_block.setZero();
    gradient_segment.setZero();
    cost = 0;
    num_subterms = 0;
  }
  Eigen::Matrix<double, kBlockDim, kBlockDim> hessian_block;
  Eigen::Matrix<double, kBlockDim, 1> gradient_segment;
  double cost;
  int num_subterms;
};

template <int kBlockDim, int kNumVarArgs>
struct LeastSquaresCostTerm {
  std::array<int, kNumVarArgs> manifold_ids;
  LeastSquaresCostTermState<kBlockDim> state;
};

template <int kBlockDim, int kNumVarArgs>
struct CostFamily {
  std::vector<LeastSquaresCostTerm<kBlockDim, kNumVarArgs>> cost_terms;
};

/// Manifold ids of the cost term arguments
template <int kArgs, class TConstArgT = farm_ng::Void>
struct CostTermRef {
  std::array<int, kArgs> arg_ids;
  TConstArgT constant;
};

template <class TArgTypes, size_t kNumArgs, size_t kI = 0>
bool constexpr areAllVarEq(
    std::array<int, kNumArgs> const& lhs,
    std::array<int, kNumArgs> const& rhs) {
  if constexpr (kI == kNumArgs) {
    return true;
  } else {
    if constexpr (std::get<kI>(TArgTypes::kArgTypeArray) == ArgType::variable) {
      if (lhs[kI] != rhs[kI]) {
        return false;
      }
    }
    return areAllVarEq<TArgTypes, kNumArgs, kI + 1>(lhs, rhs);
  }
}

template <class TArgTypes, size_t kNumArgs, size_t kI = 0>
bool constexpr lessFixed(
    std::array<int, kNumArgs> const& lhs,
    std::array<int, kNumArgs> const& rhs) {
  if constexpr (kI == kNumArgs - 1) {
    return lhs[kI] <= rhs[kI];
  } else {
    if constexpr (
        std::get<kI>(TArgTypes::kArgTypeArray) == ArgType::conditioned) {
      return lessFixed<TArgTypes, kNumArgs, kI + 1>(lhs, rhs);
    } else {
      if (lhs[kI] == rhs[kI]) {
        return lessFixed<TArgTypes, kNumArgs, kI + 1>(lhs, rhs);
      }
      return lhs[kI] < rhs[kI];
    }
  }
}

template <class TArgTypes, size_t kNumArgs, size_t kI = 0>
bool constexpr isLess(
    std::array<int, kNumArgs> const& lhs,
    std::array<int, kNumArgs> const& rhs) {
  if constexpr (kI == kNumArgs) {
    return lessFixed<TArgTypes, kNumArgs>(lhs, rhs);
  } else {
    if constexpr (std::get<kI>(TArgTypes::kArgTypeArray) == ArgType::variable) {
      return isLess<TArgTypes, kNumArgs, kI + 1>(lhs, rhs);
    } else {
      if (lhs[kI] == rhs[kI]) {
        return isLess<TArgTypes, kNumArgs, kI + 1>(lhs, rhs);
      }
      return lhs[kI] < rhs[kI];
    }
  }
}

template <bool kCalcDx, class TCostFunctor, class... TCostArg>
struct ArgTypes {
  static int constexpr kNumArgs = sizeof...(TCostArg);
  static std::array<ArgType, kNumArgs> constexpr kArgTypeArray = {
      {TCostArg::kArgType...}};
  static std::array<int, kNumArgs> constexpr kArgsDimArray =
      TCostFunctor::kArgsDimArray;

  static int constexpr kNumVarArgs = [](auto arg_type_array) {
    size_t num_vars = 0;
    for (ArgType elem : arg_type_array) {
      num_vars += elem == ArgType::variable ? 1 : 0;
    }
    return num_vars;
  }(kArgTypeArray);

  static int constexpr kDetailBlockDim = [](auto arg_type_array,
                                            auto args_dim_array) {
    size_t dim = 0;
    for (int i = 0; i < kNumArgs; ++i) {
      dim += arg_type_array[i] == ArgType::variable ? args_dim_array[i] : 0;
    }
    return dim;
  }(kArgTypeArray, kArgsDimArray);

  static int constexpr kBlockDim = kCalcDx ? kDetailBlockDim : 0;
};

template <size_t kNumArgs, size_t kI = 0, int... kInputDim, class... TManifold>
void costTermArgsFromIds(
    std::tuple<TManifold...>& cost_term_args,
    std::array<int, kNumArgs> const& arg_ids,
    std::tuple<ManifoldFamily<kInputDim, TManifold>...> const&
        manifold_family_tuple) {
  if constexpr (kI == kNumArgs) {
    return;
  } else {
    int const id = FARM_AT(arg_ids, kI);
    std::get<kI>(cost_term_args) =
        FARM_AT(std::get<kI>(manifold_family_tuple).manifolds, id);
    costTermArgsFromIds<kNumArgs, kI + 1, kInputDim...>(
        cost_term_args, arg_ids, manifold_family_tuple);
  }
}

template <class TScalar>
struct ManifoldFamilyTupleTraits;

template <int... kInputDim, class... TManifold>
struct ManifoldFamilyTupleTraits<
    std::tuple<ManifoldFamily<kInputDim, TManifold>...>> {
  using ManifoldTuple = std::tuple<TManifold...>;
};

template <
    class TArgTypes,
    size_t kNumArgs,
    size_t kI = 0,
    int... kInputDim,
    class... TManifold>
void costTermArgsFromFixedManifolds(
    std::array<int, kNumArgs>& arg_ids,
    std::tuple<TManifold...>& cost_term_args,
    [[maybe_unused]] std::array<int, TArgTypes::kNumFixedArgs> fixed_arg_ids,
    std::tuple<ManifoldFamily<kInputDim, TManifold>...> const&
        manifold_family_tuple) {
  if constexpr (kI == TArgTypes::kNumFixedArgs) {
    return;
  } else {
    static int constexpr kArgPos = std::get<kI>(TArgTypes::kFixedArgPosArray);
    int const fixed_id = FARM_AT(fixed_arg_ids, kI);
    std::get<kArgPos>(cost_term_args) =
        FARM_AT(std::get<kArgPos>(manifold_family_tuple).manifolds, fixed_id);
    std::get<kArgPos>(arg_ids) = std::get<kI>(fixed_arg_ids);
    costTermArgsFromFixedManifolds<TArgTypes, kNumArgs, kI + 1, kInputDim...>(
        arg_ids, cost_term_args, fixed_arg_ids, manifold_family_tuple);
  }
}

template <bool kCalcDx = true, class TCostFunctor, class... TCostTermArg>
static CostFamily<
    ArgTypes<kCalcDx, TCostFunctor, TCostTermArg...>::kBlockDim,
    ArgTypes<kCalcDx, TCostFunctor, TCostTermArg...>::kNumVarArgs>
apply(
    [[maybe_unused]] TCostFunctor cost_functor,
    std::vector<CostTermRef<
        ArgTypes<kCalcDx, TCostFunctor, TCostTermArg...>::kNumArgs,
        typename TCostFunctor::ConstantType>> const& arg_id_arrays,
    TCostTermArg const&... cost_arg) {
  using ArgTypesT = ArgTypes<kCalcDx, TCostFunctor, TCostTermArg...>;

  static auto constexpr kArgsDimArray = TCostFunctor::kArgsDimArray;
  static int constexpr kNumArgs = kArgsDimArray.size();
  static int constexpr kNumVarArgs = ArgTypesT::kNumVarArgs;
  static int constexpr kBlockDim = ArgTypesT::kBlockDim;

  using ConstantType = typename TCostFunctor::ConstantType;

  auto manifold_family_tuple = std::make_tuple(cost_arg.family...);
  using ManifoldFamilyTuple = decltype(manifold_family_tuple);

  CostFamily<kBlockDim, kNumVarArgs> cost_family;
  cost_family.cost_terms.reserve(arg_id_arrays.size());

  for (size_t i = 0; i < arg_id_arrays.size(); ++i) {
    auto const& arg_ids = arg_id_arrays[i].arg_ids;

    LeastSquaresCostTerm<kBlockDim, kNumVarArgs> cost_term;
    for (; i < arg_id_arrays.size(); ++i) {
      CostTermRef<kNumArgs, ConstantType> const& args = arg_id_arrays[i];

      FARM_CHECK(isLess<ArgTypesT>(arg_ids, args.arg_ids));

      typename ManifoldFamilyTupleTraits<ManifoldFamilyTuple>::ManifoldTuple
          cost_term_args;
      costTermArgsFromIds(cost_term_args, args.arg_ids, manifold_family_tuple);
      std::optional<LeastSquaresCostTermState<kBlockDim>> maybe_cost =
          std::apply(
              [&args, &cost_functor](auto... arg) {
                return cost_functor.template evalCostTerm<ArgTypesT>(
                    arg..., args.constant);
              },
              cost_term_args);

      if (!maybe_cost) {
        continue;
      }
      auto cost = FARM_UNWRAP(maybe_cost);
      cost_term.state.gradient_segment += cost.gradient_segment;
      cost_term.state.hessian_block += cost.hessian_block;
      cost_term.state.cost += cost.cost;
      cost_term.state.num_subterms += 1;

      if (!areAllVarEq<ArgTypesT>(arg_ids, args.arg_ids) ||
          i == arg_id_arrays.size() - 1) {
        break;
      }
    }
    cost_family.cost_terms.push_back(cost_term);
  }
  return cost_family;
}

}  // namespace sophus::experimental
