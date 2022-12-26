
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/concept_utils.h"

#include <Eigen/Core>

namespace sophus {

// These concept let us match Eigen's CRTP pattern and capture the true Derived
// type safely

template <class Derived>
concept EigenType = DerivedFrom<Derived, Eigen::EigenBase<Derived>>;

template <class Derived>
concept EigenDenseType = DerivedFrom<Derived, Eigen::DenseBase<Derived>>;

template <class Derived>
concept EigenMatrixType = DerivedFrom<Derived, Eigen::MatrixBase<Derived>>;

template <class Derived>
concept EigenArrayType = DerivedFrom<Derived, Eigen::ArrayBase<Derived>>;

template <class T1, typename T2>
concept EigenSameDim = requires(T1, T2) {
  EigenDenseType<T1>&& EigenDenseType<T2> &&
      (T1::RowsAtCompileTime == Eigen::Dynamic ||
       T1::RowsAtCompileTime == T2::RowsAtCompileTime) &&
      (T1::ColsAtCompileTime == Eigen::Dynamic ||
       T1::ColsAtCompileTime == T2::ColsAtCompileTime);
};

template <int Rows, int Cols, typename T>
concept EigenWithDim = requires(T) {
  EigenDenseType<T>&& T::RowsAtCompileTime == Rows&& T::ColsAtCompileTime ==
      Cols;
};

template <int Rows, int Cols, typename T>
concept EigenWithDimOrDynamic = requires(T) {
  EigenDenseType<T> &&
      (T::RowsAtCompileTime == Eigen::Dynamic ||
       T::RowsAtCompileTime == Rows) &&
      (T::ColsAtCompileTime == Eigen::Dynamic || T::ColsAtCompileTime == Cols);
};

}  // namespace sophus
