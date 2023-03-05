// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/common.h"
#include "sophus/concepts/point.h"
#include "sophus/concepts/utils.h"

namespace sophus {
namespace concepts {

template <class TT>
concept ParamsImpl = std::is_same_v<
    typename TT::Params,
    Eigen::Vector<typename TT::Scalar, TT::kNumParams>> &&
    requires(typename TT::Params params) {
  // constructors and factories
  { TT::areParamsValid(params) } -> ConvertibleTo<sophus::Expected<Success>>;

  { TT::paramsExamples() } -> ConvertibleTo<std::vector<typename TT::Params>>;

  {
    TT::invalidParamsExamples()
    } -> ConvertibleTo<std::vector<typename TT::Params>>;
};

template <class TT>
concept TangentImpl = std::is_same_v<
    typename TT::Tangent,
    Eigen::Vector<typename TT::Scalar, TT::kDof>> && requires() {
  { TT::tangentExamples() } -> ConvertibleTo<std::vector<typename TT::Tangent>>;
};

template <class TT>
concept ParamsConcept = std::is_same_v<
    typename TT::Params,
    Eigen::Vector<typename TT::Scalar, TT::kNumParams>> &&
    requires(TT m, typename TT::Params params) {
  // constructors and factories
  { TT::fromParams(params) } -> ConvertibleTo<TT>;

  {m.setParams(params)};

  { m.params() } -> ConvertibleTo<typename TT::Params>;

  { m.ptr() } -> ConvertibleTo<typename TT::Scalar const *>;

  { m.unsafeMutPtr() } -> ConvertibleTo<typename TT::Scalar *>;
};

}  // namespace concepts
}  // namespace sophus
