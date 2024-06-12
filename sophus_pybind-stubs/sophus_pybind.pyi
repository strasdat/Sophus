from __future__ import annotations
import numpy
import typing

__all__ = ["SE3", "SO3", "interpolate", "iterativeMean"]

class SE3:
    @staticmethod
    @typing.overload
    def exp(
        arg0: numpy.ndarray[numpy.float64[3, 1]],
        arg1: numpy.ndarray[numpy.float64[3, 1]],
    ) -> SE3:
        """
        Create SE3 from a translational_part (3x1) and a rotation vector (3x1) of magnitude in rad. NOTE: translational_part is not translation vector in SE3
        """
    @staticmethod
    @typing.overload
    def exp(
        arg0: numpy.ndarray[numpy.float64[m, 3]],
        arg1: numpy.ndarray[numpy.float64[m, 3]],
    ) -> SE3:
        """
        Create a set of SE3 from translational_parts (Nx3) and rotation vectors (Nx3) of magnitude in rad. NOTE: translational_part is not translation vector in SE3
        """
    @staticmethod
    @typing.overload
    def from_matrix(arg0: numpy.ndarray[numpy.float64[4, 4]]) -> SE3: ...
    @staticmethod
    @typing.overload
    def from_matrix(arg0: numpy.ndarray[numpy.float64]) -> SE3: ...
    @staticmethod
    @typing.overload
    def from_matrix3x4(arg0: numpy.ndarray[numpy.float64[3, 4]]) -> SE3: ...
    @staticmethod
    @typing.overload
    def from_matrix3x4(arg0: numpy.ndarray[numpy.float64]) -> SE3: ...
    def __copy__(self) -> SE3: ...
    def __getitem__(self, arg0: typing.Any) -> SE3: ...
    def __imatmul__(self, arg0: SE3) -> SE3: ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default Constructor initializing a group containing 1 identity element
        """
    @typing.overload
    def __init__(self, arg0: SE3) -> None:
        """
        Copy constructor from single element
        """
    def __len__(self) -> int: ...
    @typing.overload
    def __matmul__(self, arg0: SE3) -> SE3: ...
    @typing.overload
    def __matmul__(
        self, arg0: numpy.ndarray[numpy.float64[3, n]]
    ) -> numpy.ndarray[numpy.float64[3, n]]: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: typing.Any, arg1: SE3) -> None: ...
    def __str__(self) -> str: ...
    @typing.overload
    def from_quat_and_translation(
        self,
        arg0: numpy.ndarray[numpy.float64[3, 1]],
        arg1: numpy.ndarray[numpy.float64[3, 1]],
    ) -> SE3:
        """
        Create SE3 from a quaternion as w, [x, y, z], and translation vector
        """
    @typing.overload
    def from_quat_and_translation(
        self,
        arg0: numpy.ndarray[numpy.float64[m, 3]],
        arg1: numpy.ndarray[numpy.float64[m, 3]],
    ) -> SE3:
        """
        Create SE3 from a list of quaternion as w_vec: Nx1, xyz_vec: Nx3, and a list of translation vectors: Nx3
        """
    def inverse(self) -> SE3:
        """
        Compute the inverse of the transformations.
        """
    def log(self) -> numpy.ndarray[numpy.float64[m, 6]]:
        """
        Return the log of SE3 as [translational_part, rotation_vector] of dimension Nx6.
        """
    def rotation(self) -> SO3:
        """
        Get the rotation component of the transformation.
        """
    def to_matrix(self) -> numpy.ndarray[numpy.float64]:
        """
        Convert an array of SE3 into an array of transformation matrices of size 4x4
        """
    def to_matrix3x4(self) -> numpy.ndarray[numpy.float64]:
        """
        Convert an array of SE3 into an array of transformation matrices of size 3x4
        """
    def to_quat_and_translation(self) -> numpy.ndarray[numpy.float64[m, 7]]:
        """
        Return quaternion and translation as Nx7 vectors of [quat (w, x, y, z), translation]
        """
    def translation(self) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Get the translation component of the transformation.
        """

class SO3:
    @staticmethod
    def exp(arg0: numpy.ndarray[numpy.float64[m, 3]]) -> SO3:
        """
        Create rotations from rotations vectors of size Nx3 in rad
        """
    @staticmethod
    @typing.overload
    def from_matrix(arg0: numpy.ndarray[numpy.float64[3, 3]]) -> SO3: ...
    @staticmethod
    @typing.overload
    def from_matrix(arg0: numpy.ndarray[numpy.float64]) -> SO3: ...
    @staticmethod
    @typing.overload
    def from_quat(arg0: float, arg1: numpy.ndarray[numpy.float64[3, 1]]) -> SO3:
        """
        Create a rotation from a quaternion as w, [x, y, z]
        """
    @staticmethod
    @typing.overload
    def from_quat(arg0: list[float], arg1: numpy.ndarray[numpy.float64[m, 3]]) -> SO3:
        """
        Create rotations from a list of quaternions as w_vec: Nx1, xyz_vec: Nx3
        """
    def __copy__(self) -> SO3: ...
    def __getitem__(self, arg0: typing.Any) -> SO3: ...
    def __imatmul__(self, arg0: SO3) -> SO3: ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default Constructor initializing a group containing 1 identity element
        """
    @typing.overload
    def __init__(self, arg0: SO3) -> None:
        """
        Copy constructor from single element
        """
    def __len__(self) -> int: ...
    @typing.overload
    def __matmul__(self, arg0: SO3) -> SO3: ...
    @typing.overload
    def __matmul__(
        self, arg0: numpy.ndarray[numpy.float64[3, n]]
    ) -> numpy.ndarray[numpy.float64[3, n]]: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, arg0: typing.Any, arg1: SO3) -> None: ...
    def __str__(self) -> str: ...
    def inverse(self) -> SO3:
        """
        Compute the inverse of the rotations.
        """
    def log(self) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Return the rotations vector representation by taking the log operator.
        """
    def to_matrix(self) -> numpy.ndarray[numpy.float64]:
        """
        Convert an array of SO3 into an array of rotation matrices
        """
    def to_quat(self) -> numpy.ndarray[numpy.float64[m, 4]]:
        """
        Return quaternion as Nx4 vectors with the order [w x y z].
        """

def interpolate(arg0: SE3, arg1: SE3, arg2: float) -> SE3:
    """
    Interpolate two SE3s of size 1.
    """

def iterativeMean(arg0: SE3) -> SE3:
    """
    Compute the iterative mean of a sequence.
    """
