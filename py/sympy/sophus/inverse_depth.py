"""TODO(docstring)"""

import unittest

import sympy

from sophus.matrix import proj
from sophus.matrix import unproj
from sophus.matrix import vector2
from sophus.matrix import vector3
from sophus.matrix import vector6
from sophus.se3 import Isometry3


def rotation_matrix():
    """TODO(docstring)"""
    rx0, ry0, rz0 = sympy.symbols("rx[0], ry[0], rz[0]", real=True)
    rx1, ry1, rz1 = sympy.symbols("rx[1], ry[1], ry[1]", real=True)
    rx2, ry2, rz2 = sympy.symbols("rx[2], ry[2], rz[2]", real=True)
    return sympy.Matrix([[rx0, ry0, rz0], [rx1, ry1, rz1], [rx2, ry2, rz2]])


def translation():
    """TODO(docstring)"""
    t0, t1, t2 = sympy.symbols("t[0], t[1], t[2]", real=True)
    return vector3(t0, t1, t2)


class InverseDepth:
    """Affine camera transform"""

    def __init__(self, ab_and_psi):
        assert isinstance(ab_and_psi, sympy.Matrix)
        assert ab_and_psi.shape == (3, 1), ab_and_psi.shape
        self.ab_and_psi = ab_and_psi

    def scaled_transform(self, mat_r, vec_t):
        """TODO(docstring)"""
        return (
            mat_r * vector3(self.ab_and_psi[0], self.ab_and_psi[1], 1)
            + self.ab_and_psi[2] * vec_t
        )

    def dx_scaled_transform_x(self, mat_r, vec_t):
        """TODO(docstring)"""

        return sympy.Matrix(
            3,
            3,
            lambda r, c: sympy.diff(
                self.scaled_transform(mat_r, vec_t)[r], self.ab_and_psi[c]
            ).simplify(),
        )

    def calc_dx_project_transform_x(self, mat_r, vec_t):
        """TODO(docstring)"""

        return sympy.Matrix(
            2,
            3,
            lambda r, c: sympy.diff(
                proj(self.scaled_transform(mat_r, vec_t))[r], self.ab_and_psi[c]
            ).simplify(),
        )

    def dx_project_transform_x(self, mat_r, vec_t):
        """TODO(docstring)"""

        mat_rrt = sympy.Matrix([mat_r.col(0), mat_r.col(1), vec_t])

        return sympy.Matrix(
            2,
            3,
            lambda r, c: sympy.diff(
                proj(self.scaled_transform(mat_rrt, vec_t))[r], self.ab_and_psi[c]
            ).simplify(),
        )

    def dx_project_exp_x_point_at_0(self):
        """TODO(docstring)"""

        upsilon0, upsilon1, upsilon2, omega0, omega1, omega2 = sympy.symbols(
            "upsilon[0], upsilon[1], upsilon[2], omega[0], omega[1], omega[2]",
            real=True,
        )
        x = vector6(upsilon0, upsilon1, upsilon2, omega0, omega1, omega2)
        se3 = Isometry3.exp(x)
        mat_r = se3.so3.matrix()
        vec_t = se3.t

        return sympy.Matrix(
            2,
            6,
            lambda r, c: sympy.diff(
                proj(
                    mat_r * unproj(vector2(self.ab_and_psi[0], self.ab_and_psi[1]))
                    + self.ab_and_psi[2] * vec_t
                ),
                x[c],
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .subs(x[2], 0)
            .subs(x[3], 0)
            .subs(x[4], 0)
            .limit(x[5], 0),
        )

    def dx_project_exp_x_transform_at_0(self, psi_times_point):
        """TODO(docstring)"""

        upsilon0, upsilon1, upsilon2, omega0, omega1, omega2 = sympy.symbols(
            "upsilon[0], upsilon[1], upsilon[2], omega[0], omega[1], omega[2]",
            real=True,
        )
        x = vector6(upsilon0, upsilon1, upsilon2, omega0, omega1, omega2)
        se3 = Isometry3.exp(x)
        mat_r = se3.so3.matrix()
        vec_t = se3.t

        return sympy.Matrix(
            2,
            6,
            lambda r, c: sympy.diff(
                proj(mat_r * psi_times_point + self.ab_and_psi[2] * vec_t),
                x[c],
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .subs(x[2], 0)
            .subs(x[3], 0)
            .subs(x[4], 0)
            .limit(x[5], 0),
        )

    def __repr__(self):
        return "[a, b, psi]: " + repr(self.ab_and_psi)


class TestInverseDepth(unittest.TestCase):
    """TODO(docstring)"""

    def setUp(self):
        """TODO(docstring)"""

        self.mat_r = rotation_matrix()
        self.vec_t = translation()
        a, b, psi = sympy.symbols("a, b, psi", real=True)
        self.inverse_depth = InverseDepth(vector3(a, b, psi))

        x, y, z = sympy.symbols("x,y,z", real=True)
        self.xyz = vector3(x, y, z)

    def test_derivatives(self):
        """TODO(docstring)"""

        print("id_point: ", self.inverse_depth)
        print("proj: ", self.inverse_depth.scaled_transform(self.mat_r, self.vec_t))
        print(
            "calc_projectTransformRx_plus_t_x:",
            self.inverse_depth.dx_scaled_transform_x(self.mat_r, self.vec_t),
        )

        print(
            "dx_project_exp_x_point_at_0:",
            self.inverse_depth.dx_project_exp_x_point_at_0(),
        )

        print(
            "dx_project_exp_x_transform_at_0:",
            self.inverse_depth.dx_project_exp_x_transform_at_0(self.xyz),
        )

        print(
            "dx_project_transform_x:",
            self.inverse_depth.dx_project_transform_x(self.mat_r, self.vec_t),
        )


if __name__ == "__main__":
    unittest.main()
