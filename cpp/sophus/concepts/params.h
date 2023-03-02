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
concept ParamsImpl =
    requires(Eigen::Vector<typename TT::Scalar, TT::kNumParams> params) {
  // constructors and factories
  { TT::areParamsValid(params) } -> ConvertibleTo<sophus::Expected<Success>>;

  {
    TT::paramsExamples()
    } -> ConvertibleTo<
        std::vector<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>>;

  {
    TT::invalidParamsExamples()
    } -> ConvertibleTo<
        std::vector<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>>;
};

template <class TT>
concept TangentImpl = requires() {
  {
    TT::tangentExamples()
    }
    -> ConvertibleTo<std::vector<Eigen::Vector<typename TT::Scalar, TT::kDof>>>;
};

template <class TT>
concept ParamsConcept =
    requires(TT m, Eigen::Vector<typename TT::Scalar, TT::kNumParams> params) {
  // constructors and factories
  { TT::fromParams(params) } -> ConvertibleTo<TT>;

  {m.setParams(params)};

  {
    m.params()
    } -> ConvertibleTo<Eigen::Vector<typename TT::Scalar, TT::kNumParams>>;

  { m.ptr() } -> ConvertibleTo<typename TT::Scalar const *>;

  { m.unsafeMutPtr() } -> ConvertibleTo<typename TT::Scalar *>;
};

}  // namespace concepts
}  // namespace sophus
