/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "SE3PyBind.h"
#include "SO3PyBind.h"

#include <sstream>
// By default, Sophus calls std::abort when a pre-condition fails. Register a handler that raises
// an exception so we don't crash the Python process.
#ifdef SOPHUS_DISABLE_ENSURES
#undef SOPHUS_DISABLE_ENSURES
#endif
#ifndef SOPHUS_ENABLE_ENSURE_HANDLER
#define SOPHUS_ENABLE_ENSURE_HANDLER
#endif

namespace sophus {
inline void
ensureFailed(char const* function, char const* file, int line, char const* description) {
  std::stringstream message;
  message << "'SOPHUS_ENSURE' failed in function '" << function << "', on line '" << line
          << "' of file '" << file << "'. Full description:" << std::endl
          << description;
  throw std::domain_error(message.str());
}

inline void exportSophus(pybind11::module& module) {
  exportSO3Group<double>(module, "SO3");
  exportSE3Transformation<double>(module, "SE3");

  exportSE3Average<double>(module);
  exportSE3Interpolate<double>(module);
}

} // namespace sophus
