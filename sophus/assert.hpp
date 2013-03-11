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

#ifndef SOPHUS_ASSERT_HPP
#define SOPHUS_ASSERT_HPP

#include <cassert>
#include <string>

//following boost's assert.hpp
#undef SOPHUS_ASSERT

#if defined(SOPHUS_DISABLE_ASSERTS)

# define SOPHUS_ASSERT(expr, descpiption) ((void)0)

#elif defined(SOPHUS_ENABLE_ASSERT_HANDLER)

namespace Sophus {
void assertionFailed(const std::string & description);
}

#define SOPHUS_ASSERT(expr, descpiption) ((expr)                               \
  ? ((void)0)                                                                  \
  : ::Sophus::assertionFailed(descpiption))

#else
# define SOPHUS_ASSERT(expr, descpiption) ((expr)                              \
   ? ((void)0)                                                                 \
  : assert((expr)))
#endif


#endif // SOPHUS_ASSERT_HPP
