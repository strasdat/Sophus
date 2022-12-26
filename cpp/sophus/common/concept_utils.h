// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <initializer_list>

namespace sophus {

template <class TDerived, class TBase>
concept DerivedFrom = std::is_base_of_v<TBase, TDerived>;

template <class TT, class TU>
concept SameAs = std::is_same_v<TT, TU>;

template <class TT>
concept EnumType = std::is_enum_v<TT>;

template <class TT>
concept Arithmetic = std::is_arithmetic_v<TT>;

}  // namespace sophus
