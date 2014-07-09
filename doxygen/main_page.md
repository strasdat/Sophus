Sophus Lie Group Header Library {#mainpage}
====================================

Sophus is a C++ implementation of common Lie Groups and their algebra using Eigen.

Since version 0.9, Sophus has been refactored to use the Curiously Recurring
Template Pattern (CRTP), matching the design of Eigen. This gives it the
flexibility of working with Eigen::Map's and differing scalar types.

In order to go back to the non-templated/double-only version:

    git checkout a621ff

### Configuring ###

Sophus is a header-only only library and can be configured with CMake:

    cd Sophus
    mkdir build
    cd build
    cmake ..

### Using Sophus ###

Sophus exports itself automatically using CMake and can be used easily without a
Find script. To use Sophus from a CMake based project:

    ...
    find_package(Sophus)
    include_directories(${Sophus_INCLUDE_DIR})
    ...

### Documentation ###

Documentation can be built if Doxygen is installed:

    cd build
    make doc

### Testing ###

Building and running unit tests:

    cd build
    make
    ctest
