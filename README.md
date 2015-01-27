## Sophus

C++ implementation of Lie Groups using Eigen. 

### Packaging

This is a maintained version of the original code developed by Hauke Strasdat. It is also spun off as a ros 3rd party package.

### Installation

```
cd Sophus
mkdir build
cd build
cmake ..
make
```

### Errata

Thanks to Steven Lovegrove, Sophus is now fully templated  - using the Curiously Recurring Template Pattern (CRTP).

(In order to go back to the non-templated/double-only version "git checkout a621ff".)


