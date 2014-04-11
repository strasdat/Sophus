// This file is part of Sophus.
//
// Copyright 2013 Hauke Strasdat
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef SOPHUS_ENSURE_HPP
#define SOPHUS_ENSURE_HPP

#include <cstdio>
#include <cstdlib>

//following boost's assert.hpp
#undef SOPHUS_ENSURE

// ENSURES are similar to ASSERTS, but they are always checked for (including in
// release builds). At the moment there are no ASSERTS in Sophus which should
// only be used for checks which are performance critical.

#ifdef __GNUC__
#  define SOPHUS_FUNCTION __PRETTY_FUNCTION__
#elif (_MSC_VER >= 1310)
#  define SOPHUS_FUNCTION __FUNCTION__
#else
#  define SOPHUS_FUNCTION "unknown"
#endif

#if defined(SOPHUS_DISABLE_ENSURES)

#  define SOPHUS_ENSURE(expr, description) ((void)0)

#elif defined(SOPHUS_ENABLE_ENSURE_HANDLER)

namespace Sophus {
void ensureFailed(const char * function, const char * file, int line,
                  const char * description);
}

#define SOPHUS_ENSURE(expr, description) ((expr)                               \
  ? ((void)0)                                                                  \
  : ::Sophus::ensureFailed(SOPHUS_FUNCTION, __FILE__, __LINE__,                \
                           (description)))
#else
namespace Sophus {
inline
void defaultEnsure(const char * function, const char * file, int line,
                   const char * description) {
  std::printf("Sophus ensure failed in function '%s', file '%s', line %d.\n",
              function, file, line);
  std::printf("Description: %s\n",  description);
  std::abort();
}
}
#  define SOPHUS_ENSURE(expr, description) ((expr)                             \
   ? ((void)0)                                                                 \
  : Sophus::defaultEnsure(SOPHUS_FUNCTION, __FILE__, __LINE__,                 \
                          (description)))
#endif

#endif // SOPHUS_ENSURE_HPP
