---
sidebar_position: 1
---

# Sophus Style Guide

Sophus is following the [farm-ng style guide](https://farm-ng.github.io/farm-ng-core/docs/Developer%20Guide/style-guide).


## Transformations, points, velocities


## Rigid body transform and points convention

Given a point in frame ``foo``, ``bar_from_foo`` is the rigid body
transform which maps the point to frame ``bar``:

```cpp
    point_in_bar = bar_from_foo * point_in_foo
```

Note that the frame names line up: ``bar`` - ``bar``, ``foo`` - ``foo``.

### Some details

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
