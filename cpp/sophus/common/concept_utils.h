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

template <class Derived, class Base>
concept DerivedFrom = std::is_base_of_v<Base, Derived>;

template <class T, class U>
concept SameAs = std::is_same_v<T, U>;

template <class T>
concept EnumType = std::is_enum_v<T>;

template <class T>
concept Arithmetic = std::is_arithmetic_v<T>;

}  // namespace sophus
