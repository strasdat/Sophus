"""TODO(docstring)"""

import functools
import unittest

import sympy

from sophus.cse_codegen import cse_codegen
from sophus.matrix import vector3
from sophus.matrix import vector6
from sophus.matrix import zero_vector3
from sophus.matrix import zero_vector6
from sophus.matrix import proj
from sophus.matrix import squared_norm
from sophus.matrix import unproj
from sophus.quaternion import Quaternion
from sophus.so3 import Rotation3


class Isometry3:
    """3 dimensional group of rigid body transformations"""

    def __init__(self, so3, t):
        """internally represented by a unit quaternion q and a translation
        3-vector"""
        assert isinstance(so3, Rotation3)
        assert isinstance(t, sympy.Matrix)
        assert t.shape == (3, 1), t.shape

        self.so3 = so3
        self.t = t

    @staticmethod
    def exp(v):
        """exponential map"""
        upsilon = v[0:3, :]
        omega = vector3(v[3], v[4], v[5])
        so3 = Rotation3.exp(omega)
        mat_omega = Rotation3.hat(omega)
        mat_omega_sq = mat_omega * mat_omega
        theta = sympy.sqrt(squared_norm(omega))
        mat_v = (
            sympy.Matrix.eye(3)
            + (1 - sympy.cos(theta)) / (theta**2) * mat_omega
            + (theta - sympy.sin(theta)) / (theta**3) * mat_omega_sq
        )
        return Isometry3(so3, mat_v * upsilon)

    def log(self):
        """TODO(docstring)"""

        omega = self.so3.log()
        theta = sympy.sqrt(squared_norm(omega))
        mat_omega = Rotation3.hat(omega)

        half_theta = 0.5 * theta

        mat_v_inv = (
            sympy.Matrix.eye(3)
            - 0.5 * mat_omega
            + (1 - theta * sympy.cos(half_theta) / (2 * sympy.sin(half_theta)))
            / (theta * theta)
            * (mat_omega * mat_omega)
        )
        upsilon = mat_v_inv * self.t
        return upsilon.col_join(omega)

    def calc_dx_log_this(self):
        """TODO(docstring)"""
        return sympy.Matrix(6, 7, lambda r, c: sympy.diff(self.log()[r], self[c]))

    def __repr__(self):
        return "Isometry3: [" + repr(self.so3) + " " + repr(self.t)

    def inverse(self):
        """TODO(docstring)"""
        mat_r_inv = self.so3.inverse()
        return Isometry3(mat_r_inv, mat_r_inv * (-1 * self.t))

    @staticmethod
    def hat(v):
        """mat_r^6 => mat_r^4x4
        returns 4x4-matrix representation ``mat_omega``"""
        upsilon = vector3(v[0], v[1], v[2])
        omega = vector3(v[3], v[4], v[5])
        return Rotation3.hat(omega).row_join(upsilon).col_join(sympy.Matrix.zeros(1, 4))

    @staticmethod
    def vee(mat_omega):
        """mat_r^4x4 => mat_r^6
        returns 6-vector representation of Lie algebra
        This is the inverse of the hat-operator"""

        head = vector3(mat_omega[0, 3], mat_omega[1, 3], mat_omega[2, 3])
        tail = Rotation3.vee(mat_omega[0:3, 0:3])
        upsilon_omega = vector6(head[0], head[1], head[2], tail[0], tail[1], tail[2])
        return upsilon_omega

    def matrix(self):
        """returns matrix representation"""
        mat_r = self.so3.matrix()
        return (mat_r.row_join(self.t)).col_join(sympy.Matrix(1, 4, [0, 0, 0, 1]))

    def __mul__(self, right):
        """left-multiplication
        either rotation concatenation or point-transform"""
        if isinstance(right, sympy.Matrix):
            assert right.shape == (3, 1), right.shape
            return self.so3 * right + self.t
        if isinstance(right, Isometry3):
            r = self.so3 * right.so3
            t = self.t + self.so3 * right.t
            return Isometry3(r, t)
        assert False, f"unsupported type: {type(right)}"

    def __getitem__(self, key):
        """We use the following convention [q0, q1, q2, q3, t0, t1, t2]"""
        assert 0 <= key < 7
        if key < 4:
            return self.so3[key]
        return self.t[key - 4]

    @staticmethod
    def calc_dx_exp_x(x):
        """TODO(docstring)"""
        return sympy.Matrix(7, 6, lambda r, c: sympy.diff(Isometry3.exp(x)[r], x[c]))

    @staticmethod
    def dx_exp_x_at_0():
        """TODO(docstring)"""
        return sympy.Matrix(
            [
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

    def calc_dx_this_mul_exp_x_at_0(self, x):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                7, 6, lambda r, c: sympy.diff((self * Isometry3.exp(x))[r], x[c])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .subs(x[2], 0)
            .subs(x[3], 0)
            .subs(x[4], 0)
            .limit(x[5], 0)
        )

    @staticmethod
    def calc_dx_exp_x_at_0(x):
        """TODO(docstring)"""
        return (
            Isometry3.calc_dx_exp_x(x)
            .subs(x[0], 0)
            .subs(x[1], 0)
            .subs(x[2], 0)
            .subs(x[3], 0)
            .subs(x[4], 0)
            .limit(x[5], 0)
        )

    @staticmethod
    def dxi_x_matrix(x, i):
        """TODO(docstring)"""
        if i < 4:
            return (
                Rotation3.dxi_x_matrix(x, i)
                .row_join(sympy.Matrix.zeros(3, 1))
                .col_join(sympy.Matrix.zeros(1, 4))
            )
        mat_m = sympy.Matrix.zeros(4, 4)
        mat_m[i - 4, 3] = 1
        return mat_m

    @staticmethod
    def calc_dxi_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(4, 4, lambda r, c: sympy.diff(x.matrix()[r, c], x[i]))

    @staticmethod
    def dxi_exp_x_matrix(x, i):
        """TODO(docstring)"""
        mat_t = Isometry3.exp(x)
        dx_exp_x = Isometry3.calc_dx_exp_x(x)
        l = [dx_exp_x[j, i] * Isometry3.dxi_x_matrix(mat_t, j) for j in range(0, 7)]
        return functools.reduce((lambda a, b: a + b), l)

    @staticmethod
    def calc_dxi_exp_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(
            4, 4, lambda r, c: sympy.diff(Isometry3.exp(x).matrix()[r, c], x[i])
        )

    @staticmethod
    def dxi_exp_x_matrix_at_0(i):
        """TODO(docstring)"""
        v = zero_vector6()
        v[i] = 1
        return Isometry3.hat(v)

    @staticmethod
    def calc_dxi_exp_x_matrix_at_0(x, i):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                4, 4, lambda r, c: sympy.diff(Isometry3.exp(x).matrix()[r, c], x[i])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .subs(x[2], 0)
            .subs(x[3], 0)
            .subs(x[4], 0)
            .limit(x[5], 0)
        )


class TestSe3(unittest.TestCase):
    """TODO(docstring)"""

    # pylint: disable=too-many-locals
    def setUp(self):
        upsilon0, upsilon1, upsilon2, omega0, omega1, omega2 = sympy.symbols(
            "upsilon[0], upsilon[1], upsilon[2], omega[0], omega[1], omega[2]",
            real=True,
        )
        x, v0, v1, v2 = sympy.symbols("q.w() q.x() q.y() q.z()", real=True)
        p0, p1, p2 = sympy.symbols("p0 p1 p2", real=True)
        t0, t1, t2 = sympy.symbols("t[0] t[1] t[2]", real=True)
        v = vector3(v0, v1, v2)
        self.upsilon_omega = vector6(
            upsilon0, upsilon1, upsilon2, omega0, omega1, omega2
        )
        self.t = vector3(t0, t1, t2)
        self.a = Isometry3(Rotation3(Quaternion(x, v)), self.t)
        self.p = vector3(p0, p1, p2)

    def test_exp_log(self):
        """TODO(docstring)"""
        for v in [
            vector6(0.0, 1, 0.5, 2.0, 1, 0.5),
            vector6(0.1, 0.1, 0.1, 0.0, 1, 0.5),
            vector6(0.01, 0.2, 0.03, 0.01, 0.2, 0.03),
        ]:
            w = Isometry3.exp(v).log()
            for i in range(0, 3):
                self.assertAlmostEqual(v[i], w[i])

    def test_matrix(self):
        """TODO(docstring)"""
        foo_from_bar = Isometry3.exp(self.upsilon_omega)
        foo_transform_bar = foo_from_bar.matrix()
        point_bar = self.p
        p1_foo = foo_from_bar * point_bar
        p2_foo = proj(foo_transform_bar * unproj(point_bar))
        self.assertEqual(sympy.simplify(p1_foo - p2_foo), zero_vector3())

    def test_derivatives(self):
        """TODO(docstring)"""
        self.assertEqual(
            sympy.simplify(
                Isometry3.calc_dx_exp_x_at_0(self.upsilon_omega)
                - Isometry3.dx_exp_x_at_0()
            ),
            sympy.Matrix.zeros(7, 6),
        )

        for i in range(0, 7):
            self.assertEqual(
                sympy.simplify(
                    Isometry3.calc_dxi_x_matrix(self.a, i)
                    - Isometry3.dxi_x_matrix(self.a, i)
                ),
                sympy.Matrix.zeros(4, 4),
            )
        for i in range(0, 6):
            self.assertEqual(
                sympy.simplify(
                    Isometry3.dxi_exp_x_matrix(self.upsilon_omega, i)
                    - Isometry3.calc_dxi_exp_x_matrix(self.upsilon_omega, i)
                ),
                sympy.Matrix.zeros(4, 4),
            )
            self.assertEqual(
                sympy.simplify(
                    Isometry3.dxi_exp_x_matrix_at_0(i)
                    - Isometry3.calc_dxi_exp_x_matrix_at_0(self.upsilon_omega, i)
                ),
                sympy.Matrix.zeros(4, 4),
            )

    def test_codegen(self):
        """TODO(docstring)"""
        stream = cse_codegen(self.a.calc_dx_exp_x(self.upsilon_omega))
        filename = "cpp_gencode/Se3_Dx_exp_x.cpp"
        # set to true to generate codegen files
        # pylint: disable=using-constant-test
        if False:
            with open(filename, "w", encoding="utf-8") as file:
                for line in stream:
                    file.write(line)
        else:
            with open(filename, "r", encoding="utf-8") as file:
                file_lines = file.readlines()
                for i, line in enumerate(stream):
                    self.assertEqual(line, file_lines[i])

        stream = cse_codegen(self.a.calc_dx_this_mul_exp_x_at_0(self.upsilon_omega))
        filename = "cpp_gencode/Se3_Dx_this_mul_exp_x_at_0.cpp"
        # set to true to generate codegen files
        # pylint: disable=using-constant-test
        if False:
            with open(filename, "w", encoding="utf-8") as file:
                for line in stream:
                    file.write(line)
        else:
            with open(filename, "r", encoding="utf-8") as file:
                file_lines = file.readlines()
                for i, line in enumerate(stream):
                    self.assertEqual(line, file_lines[i])

        stream = cse_codegen(self.a.calc_dx_log_this())
        filename = "cpp_gencode/Se3_Dx_log_this.cpp"

        # set to true to generate codegen files
        # pylint: disable=using-constant-test
        if False:
            with open(filename, "w", encoding="utf-8") as file:
                for line in stream:
                    file.write(line)
        else:
            with open(filename, "r", encoding="utf-8") as file:
                file_lines = file.readlines()
                for i, line in enumerate(stream):
                    self.assertEqual(line, file_lines[i])


if __name__ == "__main__":
    unittest.main()
