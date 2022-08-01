#!/bin/bash

set -x # echo on
set -e # exit on error

brew update
brew install fmt


# Build a specific version of ceres-solver instead of one shipped over brew
curl https://raw.githubusercontent.com/Homebrew/homebrew-core/b0792ccba6e71cd028263ca7621db894afc602d2/Formula/ceres-solver.rb -o ceres-solver.rb
patch <<EOF
--- ceres-solver.rb
+++ ceres-solver.rb
@@ -1,8 +1,8 @@
 class CeresSolver < Formula
   desc "C++ library for large-scale optimization"
   homepage "http://ceres-solver.org/"
-  url "http://ceres-solver.org/ceres-solver-2.0.0.tar.gz"
-  sha256 "10298a1d75ca884aa0507d1abb0e0f04800a92871cd400d4c361b56a777a7603"
+  url "https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0rc1.tar.gz"
+  sha256 "9138a7d80a3142fe3a98519d58a489da9e204b815cd8c0571b3e643f04eca574"
   license "BSD-3-Clause"
   revision 4
   head "https://ceres-solver.googlesource.com/ceres-solver.git", branch: "master"
@@ -31,23 +31,16 @@
   depends_on "suite-sparse"
   depends_on "tbb"
 
-  # Fix compatibility with TBB 2021.1
-  # See https://github.com/ceres-solver/ceres-solver/issues/669
-  # Remove in the next release
-  patch do
-    url "https://github.com/ceres-solver/ceres-solver/commit/941ea13475913ef8322584f7401633de9967ccc8.patch?full_index=1"
-    sha256 "c61ca2ff1e92cc2134ba8e154bd9052717ba3fcae085e8f44957b9c22e6aa4ff"
-  end
 
   def install
     system "cmake", ".", *std_cmake_args,
                     "-DBUILD_SHARED_LIBS=ON",
                     "-DBUILD_EXAMPLES=OFF",
-                    "-DLIB_SUFFIX=''"
+                    "-DLIB_SUFFIX=''",
+                    "-DCMAKE_CXX_COMPILER_LAUNCHER=/usr/local/bin/ccache"
     system "make"
     system "make", "install"
     pkgshare.install "examples", "data"
-    doc.install "docs/html" unless build.head?
   end
EOF

HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1 brew install --build-from-source ./ceres-solver.rb
