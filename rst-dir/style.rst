
C++ Coding Style
================

Coding guidelines
-----------------

We aim for using modern c++. A general good resource for best practice are the
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md.


Code formating
--------------

We use clang-format. See ``.clang-format`` file in the root directory for
details.

Headers
-------

Each header file shall start with a ``#pragma once`` statement (i.e. right after
the copyright / licensing statement).

Header include order
********************

Header includes should be ordered in groups, proceeding from "local" to "global".
This `helps ensure <https://stackoverflow.com/questions/2762568>`_ that local
headers are self-sufficient, without implicit dependencies. This convention is
enforced by `clang-format`.

When using C library header one shall prefer C++ library version, over the
C-library version: ``#include <cmath>`` instead of ``#include <math.h>``.


Naming convention
-----------------

Most naming conventions are enforced by clang-tidy. See ``.clang-tidy`` in the
root for details.

``snake_case``
**************

Variable names are typically written in ``snake_case``, e.g.::

  int count = 0;
  std::vector<double> floating_point_numbers = {0.1, 0.3};
  size_t num_numbers = floating_point_numbers.size();


``snake_case`` is also used for namespaces, see below.


``trailing_snake_case_``
************************

Private member variables of classes are written in ``trailing_snake_case_``,
e.g.::

    class Foo {
     public:
      /* details */

     private:
      std::vector<int> integers_;
      Bar internal_state_;
    }


``PascalCase``
**************

Type names, including class names, struct names and typedefs, are written in
``PascalCase``::

    struct Foo {};

    class Bar {
      /* details */
    };

    using FooBarVariant = std::variant<Foo, Bar>;

``lowerCamelCase``
******************

Function, including free functions and class methods are written in
``lowerCamelCase``::

    void solveSubProblem(const Input& input);

    class Bar {
     public:
      DebugOutput getDebugOutput();
    };

    Foo fooFromBar(const Bar& bar);

Note that lambda expressions names are considered variables and hence we use
``snake_case``::

    auto solve_sub_problem = [&input]() { return solveSubProblem(input); }

``kLowerCamelCase`` is used for global and static class constants, constexpr
constants and value class templates see below.


``UPPER_CASE``
**************

Macros only. See below.


Acronyms
********

Acronyms and contractions count as one word. That is we have ``SlamResult``,
``HttpRequest`` and ``Se3Group`` (and not ``SLAMResult``, ``HTTPRequest``,
``SE3Group``) in UpperCamelCase and ``slam_result`` and ``http_request`` in
snake_case.

Classes and structs
*******************

While C++ allows for a great flexibility, we impose some guidelines when to use
structs vs. classes.

For POD-like collections we use the struct keyword::

    struct Collection {
      std::vector<int> integers;
      std::string description;
    };

Such structs must not have private members. They may or may not have have member
functions, and constructors. For the public member variables we use
``snake_case`` naming.

For entities with non-trivial type invariant, we use the class keyword::

    template <class T>
    class UnitVector3 {
     public:
      // Precondition: v must be of unit length.
      static UnitVector3 fromUnitVector(const Eigen::Matrix<T, 3, 1>& v) {
        using std::abs;
        FARM_CHECK_LE((v.squaredNorm() - T(1.0)),
                         Sophus::kEpsilon<TT>);
        UnitVector3 unit_vector;
        unit_vector.vector_ = v;
        return unit_vector;
      }


      static UnitVector3 fromVectorAndNormalize(
          const Eigen::Matrix<T, 3, 1>& v) {
        return fromUnitVector(v.normalized());
      }

      const Eigen::Matrix<T, 3, 1>& vector() const { return vector_; }

     private:
      UnitVector3() {}

      // Class invariant: v_ is of unit length.
      Eigen::Matrix<T, 3, 1> vector_;
    };

Classes shall start with all public member (constructors, destructors, member
functions etc.) then followed by private members (member functions then member
variables). Classes shall not have any public member variables. Use public
accessors/mutators instead::

    class MyClass {
    public:
      /* details */

      // accessor
      [[nodiscard]] const std::vector<double>& rawValues() const {
        return raw_values_;
      }

      // mutator
      std::vector<double>& mutRawValues() {
        return raw_values_;
      }


    private:
      std::vector<double> raw_values_;
    };

Notes about class/struct methods:

 - Do not prefix an accessor with ``get``.
 - It is good practice to annotate a constant ref accessor with
   ``[[nodiscard]]``.
 - It is recommended to prefix mutators and other non-const methods with
   ``mut``. This is inspired by the ``mut`` postfix convention in rust (e.g.
   https://doc.rust-lang.org/std/vec/struct.Vec.html#method.last_mut) and it is
   similar to the rational of the introduction of ``cbegin/cend`` to the c++
   standard (as motivated here
   https://stackoverflow.com/questions/12001410/what-is-the-reason-behind-cbegin-cend).


For implementation details, hence code not part of a public API, such as trait
classes, ceres::Cost functors, Impl classes (e.g. when using the Pimpl
idiom: http://bitboom.github.io/pimpl-idiom), it is fine to a have a
class/struct with all public member variables.


Enums
*****

Prefer use enum classes defined through the FARM_ENUM macro.

Example::

    FARM_ENUM(VizMaterialType, (ambient, diffuse, phong));


Note: we use ``snake_case`` for enum value names, such that the
corresponding strings are more user-friendly, e.g. when passing
in values through CLI11::

    std::string example_input = "ambient";
    VizMatrialType material;
    FARM_CHECK(trySetFromString(material, example_input));


Constants
*********

For global and static class constants, we use the ``kLowerCamelCase`` naming
conventions. Examples::

    constexpr double kPi = 3.14159265359;

    class Variable5Dof {
     public:
      static constexpr int kNumDegreesOfFreedom = 5;

      /* details */
    };

Macros
******

Avoid using macros, especially if the same behaviors can be achieved through
constants or C++ templates.

For all marcos we use the ``FARM_UPPER_CASE`` naming style. In particular,
all macro names shall be prefixed by ``FARM_``. Example::

    #define FARM_FORMAT(cstr, ...)                             \
        /* FMT_STRING is defined in the <fmt/format.h> header */  \
        ::fmt::format(FMT_STRING(cstr), __VA_ARGS__)


Namespaces
**********

For namespace hierarchy, we believe more is less. That is most of the code shall
be defined within the top-level namespace, e.g. ``sophus`` for the Sophus
library.

For implementation details (e.g. in header only libraries), we use the
``sophus::details`` namespace.

All namespace names are in ``snake_case``.


Templates
*********

Use reasonable. Compile time matters too.


`class` versus `typename` in templates
--------------------------------------

We use `class`.

Both keywords are interchangeable here:

```
template<class T>
class Foo {
};
and
```

and

```
template<typename T>
class Foo {
};
```
(https://stackoverflow.com/a/2024173).

W use the `class` key name in (class, function, variable) templates
always:

 - to be consistent,
 - `class` is less letters to type than `typename`,
 - possibly easier to parse in complex expressions
   (`class` looks less similar to `template` keyword).


Transformations, points, velocities
-----------------------------------

Rigid body transform and points convention
******************************************

Given a point in frame ``foo``, ``bar_from_foo`` is the rigid body
transform which maps the point to frame ``bar``:

    point_in_bar = bar_from_foo * point_in_foo

Note that the frame names line up: ``bar`` - ``bar``, ``foo`` - ``foo``.

Some details
************

 - If necessary to resolved ambiguities, e.g. between rotations and rigid
   body transforms, we use a corresponding notation, e.g.
   ``bar_form_foo_rotation``, ``bar_form_foo_se3s`` etc.

 - If we have a list (or vector, or map) of transforms we write:, e.g.:

      bar_from_foo_rotations

 - Transforms with compound frame names, such as ``robot_base`` or
   ``left_camera``, are written as follows::

      robot_base_from_left_camera

 - For functions and methods we use ``camelCase``. Examples::

      this->calcRobotBaseFromLeftCamera();

      other->setBarFromFoo(bar_from_foo);

 - We use the ``entity_in_frame`` conventions for points and other entities
   which have one frame attached to it. Examples:

   * ``point_in_camera``
   * ``circle_in_image``
   * ``camera_position_in_world`` (or short ``camera_in_world``)
   * ...

  - When storing transforms / using them in interfaces, prefer the
    ``parent_from_child`` convention.

    E.g. ``world_from_sensor_rig``, ``sensor_rig_from_camera``,
    ``robot_from_imu`` (and not ``sensor_rig_from_world`` etc.).

    It is easier to reason about the pose of camera in the world frame, then
    the pose of the world origin in the camera frame.


Some geeky notes about the terms poses, orientation and position
================================================================

* Note: This section can be safely skipped - and is merely targeting notation
geeks.

The notation and terms above (and in Sophus in general) is of a functional
nature - describing motions/actions of rigid bodies in space.

For instance, ``world_from_camera_transform`` describes the rotations followed
by a translation to be applied to points in the camera reference frame in order
to map them with the corresponding set of points in the world frame.

This motion is mainly metaphorically as it merely a way to describe the
structural relationship between the world reference frame and the camera frame.

An alternative and equivalent terminology follows a *predicative* description,
describing structural relationship directly.

Instead of (rigid body) transformation, one would use the term (rigid body)
pose. Similarly instead of rotation (transformations) and translation
(transformations), one would use the terms orientation and position.

The intro of this chapter would be written as such:

  bar_anchored_foo_pose is the rigid body transform which relates points in
  frame foo to points in frame bar such as:

      point_in_bar = bar_anchored_foo_pose * point_in_foo

  Note that the frame names line up: bar - bar, foo - foo.


In Sophus, function, variable names etc. follow the *functional* convention
using terms such as rotation and translation.

However, some concepts are easier understood using a *predicative* description
 - hence in comments and descriptions, predicative terms such as pose and
 position are used when it may facilitate explaining the underlying concept.

For instance, depending on the context we may prefer

  ``world_from_camera_rotation`` describes the orientation of camera,
  with respect to the world reference frame.

over

  ``world_from_camera_rotation`` is the rotation transformation, which
  maps a set of points in the camera camera, to corresponding point set
  in the world reference frame.


[See Schwank: "Cognitive Structures and Cognitive Strategies in Algorithmic
Thinking", https://link.springer.com/chapter/10.1007/978-3-662-11334-9_22 for
background on functional vs. predicative thinking.]


Angular and linear velocities
-----------------------------

While sometimes it is useful to express velocities with respect to to three
frames (see http://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly),
most of the time two frames are sufficient.

For example, for the linear velocity *of* the ``IMU`` module *as seen from* and
*expressed in* the ``world`` frame, we use::

    world_from_imu_velocity

For the corresponding angular rate (or rotational velocity), we write::

    world_from_imu_angular_rate


If there is a use-case where three frames are involved, as in the angular
rate *of* frame C *as seen from* frame B, *expressed in* frame A, we use::

    a_from_b_of_c_angular_rate
