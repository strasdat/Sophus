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

template <int kTangentDim, class ManifoldT = Eigen::Vector<double, kTangentDim>>
struct ManifoldFamily {
  using Manifold = ManifoldT;

  std::vector<Manifold> manifolds;

  Manifold oplus(
      Manifold const& g, Eigen::Vector<double, kTangentDim> const& vec_a);
};

template <int kTangentDim, class ManifoldT>
struct Var {
  static ArgType constexpr kArgType = ArgType::variable;

  Var(ManifoldFamily<kTangentDim, ManifoldT> const& family) : family(family) {}
  ManifoldFamily<kTangentDim, ManifoldT> const& family;
};

template <int kTangentDim, class ManifoldT>
struct CondVar {
  static ArgType constexpr kArgType = ArgType::conditioned;

  CondVar(ManifoldFamily<kTangentDim, ManifoldT> const& family)
      : family(family) {}
  ManifoldFamily<kTangentDim, ManifoldT> const& family;
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
  double cost = 0;
  int num_subterms = 0;
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
template <int kArgs, class ConstArgT = farm_ng::Void>
struct CostTermRef {
  std::array<int, kArgs> arg_ids;
  ConstArgT constant;
};

template <class ArgTypesT, size_t kNumArgs, size_t kI = 0>
constexpr bool areAllVarEq(
    std::array<int, kNumArgs> const& lhs,
    std::array<int, kNumArgs> const& rhs) {
  if constexpr (kI == kNumArgs) {
    return true;
  } else {
    if constexpr (std::get<kI>(ArgTypesT::kArgTypeArray) == ArgType::variable) {
      if (lhs[kI] != rhs[kI]) {
        return false;
      }
    }
    return areAllVarEq<ArgTypesT, kNumArgs, kI + 1>(lhs, rhs);
  }
}

template <class ArgTypesT, size_t kNumArgs, size_t kI = 0>
constexpr bool lessFixed(
    std::array<int, kNumArgs> const& lhs,
    std::array<int, kNumArgs> const& rhs) {
  if constexpr (kI == kNumArgs - 1) {
    return lhs[kI] <= rhs[kI];
  } else {
    if constexpr (
        std::get<kI>(ArgTypesT::kArgTypeArray) == ArgType::conditioned) {
      return lessFixed<ArgTypesT, kNumArgs, kI + 1>(lhs, rhs);
    } else {
      if (lhs[kI] == rhs[kI]) {
        return lessFixed<ArgTypesT, kNumArgs, kI + 1>(lhs, rhs);
      }
      return lhs[kI] < rhs[kI];
    }
  }
}

template <class ArgTypesT, size_t kNumArgs, size_t kI = 0>
constexpr bool isLess(
    std::array<int, kNumArgs> const& lhs,
    std::array<int, kNumArgs> const& rhs) {
  if constexpr (kI == kNumArgs) {
    return lessFixed<ArgTypesT, kNumArgs>(lhs, rhs);
  } else {
    if constexpr (std::get<kI>(ArgTypesT::kArgTypeArray) == ArgType::variable) {
      return isLess<ArgTypesT, kNumArgs, kI + 1>(lhs, rhs);
    } else {
      if (lhs[kI] == rhs[kI]) {
        return isLess<ArgTypesT, kNumArgs, kI + 1>(lhs, rhs);
      }
      return lhs[kI] < rhs[kI];
    }
  }
}

template <bool kCalcDx, class CostFunctorT, class... CostArgT>
struct ArgTypes {
  static int constexpr kNumArgs = sizeof...(CostArgT);
  static std::array<ArgType, kNumArgs> constexpr kArgTypeArray = {
      {CostArgT::kArgType...}};
  static std::array<int, kNumArgs> constexpr kArgsDimArray =
      CostFunctorT::kArgsDimArray;

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

template <size_t kNumArgs, size_t kI = 0, int... kInputDim, class... ManifoldT>
void costTermArgsFromIds(
    std::tuple<ManifoldT...>& cost_term_args,
    std::array<int, kNumArgs> const& arg_ids,
    std::tuple<ManifoldFamily<kInputDim, ManifoldT>...> const&
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

template <class TT>
struct ManifoldFamilyTupleTraits;

template <int... kInputDim, class... ManifoldT>
struct ManifoldFamilyTupleTraits<
    std::tuple<ManifoldFamily<kInputDim, ManifoldT>...>> {
  using ManifoldTuple = std::tuple<ManifoldT...>;
};

template <
    class ArgTypesT,
    size_t kNumArgs,
    size_t kI = 0,
    int... kInputDim,
    class... ManifoldT>
void costTermArgsFromFixedManifolds(
    std::array<int, kNumArgs>& arg_ids,
    std::tuple<ManifoldT...>& cost_term_args,
    [[maybe_unused]] std::array<int, ArgTypesT::kNumFixedArgs> fixed_arg_ids,
    std::tuple<ManifoldFamily<kInputDim, ManifoldT>...> const&
        manifold_family_tuple) {
  if constexpr (kI == ArgTypesT::kNumFixedArgs) {
    return;
  } else {
    static int constexpr kArgPos = std::get<kI>(ArgTypesT::kFixedArgPosArray);
    int const fixed_id = FARM_AT(fixed_arg_ids, kI);
    std::get<kArgPos>(cost_term_args) =
        FARM_AT(std::get<kArgPos>(manifold_family_tuple).manifolds, fixed_id);
    std::get<kArgPos>(arg_ids) = std::get<kI>(fixed_arg_ids);
    costTermArgsFromFixedManifolds<ArgTypesT, kNumArgs, kI + 1, kInputDim...>(
        arg_ids, cost_term_args, fixed_arg_ids, manifold_family_tuple);
  }
}

template <bool kCalcDx = true, class CostFunctorT, class... CostTermArgT>
static CostFamily<
    ArgTypes<kCalcDx, CostFunctorT, CostTermArgT...>::kBlockDim,
    ArgTypes<kCalcDx, CostFunctorT, CostTermArgT...>::kNumVarArgs>
apply(
    [[maybe_unused]] CostFunctorT cost_functor,
    std::vector<CostTermRef<
        ArgTypes<kCalcDx, CostFunctorT, CostTermArgT...>::kNumArgs,
        typename CostFunctorT::ConstantType>> const& arg_id_arrays,
    CostTermArgT const&... cost_arg) {
  using ArgTypesT = ArgTypes<kCalcDx, CostFunctorT, CostTermArgT...>;

  static auto constexpr kArgsDimArray = CostFunctorT::kArgsDimArray;
  static int constexpr kNumArgs = kArgsDimArray.size();
  static int constexpr kNumVarArgs = ArgTypesT::kNumVarArgs;
  static int constexpr kBlockDim = ArgTypesT::kBlockDim;

  using ConstantType = typename CostFunctorT::ConstantType;

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
