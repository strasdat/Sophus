linux, os x: [![Build Status](https://travis-ci.org/strasdat/Sophus.svg?branch=master)](https://travis-ci.org/strasdat/Sophus) 
windows: [![Build status](https://ci.appveyor.com/api/projects/status/um4285lwhs8ci7pt/branch/master?svg=true)](https://ci.appveyor.com/project/strasdat/sophus/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/strasdat/Sophus/badge.svg?branch=master)](https://coveralls.io/github/strasdat/Sophus?branch=master)

Sophus
------

C++ implementation of Lie Groups using Eigen.

Thanks to Steven Lovegrove, Sophus is now fully templated  - using the Curiously Recurring Template Pattern (CRTP).

(In order to go back to the non-templated/double-only version "git checkout a621ff".)

Installation guide:

```
cd Sophus
mkdir build
cd build
cmake ..
make
```


