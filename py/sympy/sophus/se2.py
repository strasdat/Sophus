"""TODO(docstring)"""

import functools
import unittest

import sympy

from sophus.complex import Complex
from sophus.cse_codegen import cse_codegen
from sophus.matrix import vector2
from sophus.matrix import vector3
from sophus.matrix import zero_vector2
from sophus.matrix import zero_vector3
from sophus.matrix import proj
from sophus.matrix import unproj
from sophus.so2 import Rotation2


class Isometry2:
    """2 dimensional group of rigid body transformations"""

    def __init__(self, so2, t):
        """internally represented by a unit complex number z and a translation
        2-vector"""
        self.so2 = so2
        self.t = t

    @staticmethod
    def exp(v):
        """exponential map"""
        theta = v[2]
        so2 = Rotation2.exp(theta)

        a = so2.z.imag / theta
        b = (1 - so2.z.real) / theta

        t = vector2(a * v[0] - b * v[1], b * v[0] + a * v[1])
        return Isometry2(so2, t)

    def log(self):
        """TODO(docstring)"""
        theta = self.so2.log()
        halftheta = 0.5 * theta
        a = -(halftheta * self.so2.z.imag) / (self.so2.z.real - 1)

        mat_v_inv = sympy.Matrix([[a, halftheta], [-halftheta, a]])
        upsilon = mat_v_inv * self.t
        return vector3(upsilon[0], upsilon[1], theta)

    def calc_dx_log_this(self):
        """TODO(docstring)"""
        return sympy.Matrix(3, 4, lambda r, c: sympy.diff(self.log()[r], self[c]))

    def __repr__(self):
        return "Isometry2: [" + repr(self.so2) + " " + repr(self.t)

    @staticmethod
    def hat(v):
        """TODO(docstring)"""
        upsilon = vector2(v[0], v[1])
        theta = v[2]
        return Rotation2.hat(theta).row_join(upsilon).col_join(sympy.Matrix.zeros(1, 3))

    def matrix(self):
        """returns matrix representation"""
        mat_r = self.so2.matrix()
        return (mat_r.row_join(self.t)).col_join(sympy.Matrix(1, 3, [0, 0, 1]))

    def __mul__(self, right):
        """left-multiplication
        either rotation concatenation or point-transform"""
        if isinstance(right, sympy.Matrix):
            assert right.shape == (2, 1), right.shape
            return self.so2 * right + self.t
        if isinstance(right, Isometry2):
            return Isometry2(self.so2 * right.so2, self.t + self.so2 * right.t)
        assert False, f"unsupported type: {type(right)}"

    def __getitem__(self, key):
        """We use the following convention [q0, q1, q2, q3, t0, t1, t2]"""
        assert 0 <= key < 4
        if key < 2:
            return self.so2[key]
        return self.t[key - 2]

    @staticmethod
    def calc_dx_exp_x(x):
        """TODO(docstring)"""
        return sympy.Matrix(4, 3, lambda r, c: sympy.diff(Isometry2.exp(x)[r], x[c]))

    @staticmethod
    def dx_exp_x_at_0():
        """TODO(docstring)"""
        return sympy.Matrix([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def calc_dx_this_mul_exp_x_at_0(self, x):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                4, 3, lambda r, c: sympy.diff((self * Isometry2.exp(x))[r], x[c])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .limit(x[2], 0)
        )

    @staticmethod
    def calc_dx_exp_x_at_0(x):
        """TODO(docstring)"""
        return Isometry2.calc_dx_exp_x(x).subs(x[0], 0).subs(x[1], 0).limit(x[2], 0)

    @staticmethod
    def dxi_x_matrix(x, i):
        """TODO(docstring)"""
        if i < 2:
            return (
                Rotation2.dxi_x_matrix(x, i)
                .row_join(sympy.Matrix.zeros(2, 1))
                .col_join(sympy.Matrix.zeros(1, 3))
            )
        mat_m = sympy.Matrix.zeros(3, 3)
        mat_m[i - 2, 2] = 1
        return mat_m

    @staticmethod
    def calc_dxi_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(3, 3, lambda r, c: sympy.diff(x.matrix()[r, c], x[i]))

    @staticmethod
    def dxi_exp_x_matrix(x, i):
        """TODO(docstring)"""
        mat_t = Isometry2.exp(x)
        dx_exp_x = Isometry2.calc_dx_exp_x(x)
        l = [dx_exp_x[j, i] * Isometry2.dxi_x_matrix(mat_t, j) for j in range(0, 4)]
        return functools.reduce((lambda a, b: a + b), l)

    @staticmethod
    def calc_dxi_exp_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(
            3, 3, lambda r, c: sympy.diff(Isometry2.exp(x).matrix()[r, c], x[i])
        )

    @staticmethod
    def dxi_exp_x_matrix_at_0(i):
        """TODO(docstring)"""
        v = zero_vector3()
        v[i] = 1
        return Isometry2.hat(v)

    @staticmethod
    def calc_dxi_exp_x_matrix_at_0(x, i):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                3, 3, lambda r, c: sympy.diff(Isometry2.exp(x).matrix()[r, c], x[i])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .limit(x[2], 0)
        )


class TestSe2(unittest.TestCase):
    """TODO(docstring)"""

    # pylint: disable=too-many-locals
    def setUp(self):
        upsilon0, upsilon1, theta = sympy.symbols(
            "upsilon[0], upsilon[1], theta", real=True
        )
        x, y = sympy.symbols("c[0] c[1]", real=True)
        p0, p1 = sympy.symbols("p0 p1", real=True)
        t0, t1 = sympy.symbols("t[0] t[1]", real=True)
        self.upsilon_theta = vector3(upsilon0, upsilon1, theta)
        self.t = vector2(t0, t1)
        self.a = Isometry2(Rotation2(Complex(x, y)), self.t)
        self.p = vector2(p0, p1)

    def test_exp_log(self):
        """TODO(docstring)"""
        for v in [
            vector3(0.0, 1, 0.5),
            vector3(0.1, 0.1, 0.1),
            vector3(0.01, 0.2, 0.03),
        ]:
            w = Isometry2.exp(v).log()
            for i in range(0, 3):
                self.assertAlmostEqual(v[i], w[i])

    def test_matrix(self):
        """TODO(docstring)"""
        foo_from_bar = Isometry2.exp(self.upsilon_theta)
        foo_transform_bar = foo_from_bar.matrix()
        point_bar = self.p
        p1_foo = foo_from_bar * point_bar
        p2_foo = proj(foo_transform_bar * unproj(point_bar))
        self.assertEqual(sympy.simplify(p1_foo - p2_foo), zero_vector2())

    def test_derivatives(self):
        """TODO(docstring)"""
        self.assertEqual(
            sympy.simplify(
                Isometry2.calc_dx_exp_x_at_0(self.upsilon_theta)
                - Isometry2.dx_exp_x_at_0()
            ),
            sympy.Matrix.zeros(4, 3),
        )
        for i in range(0, 4):
            self.assertEqual(
                sympy.simplify(
                    Isometry2.calc_dxi_x_matrix(self.a, i)
                    - Isometry2.dxi_x_matrix(self.a, i)
                ),
                sympy.Matrix.zeros(3, 3),
            )
        for i in range(0, 3):
            self.assertEqual(
                sympy.simplify(
                    Isometry2.dxi_exp_x_matrix(self.upsilon_theta, i)
                    - Isometry2.calc_dxi_exp_x_matrix(self.upsilon_theta, i)
                ),
                sympy.Matrix.zeros(3, 3),
            )
            self.assertEqual(
                sympy.simplify(
                    Isometry2.dxi_exp_x_matrix_at_0(i)
                    - Isometry2.calc_dxi_exp_x_matrix_at_0(self.upsilon_theta, i)
                ),
                sympy.Matrix.zeros(3, 3),
            )

    def test_codegen(self):
        """TODO(docstring)"""
        stream = cse_codegen(Isometry2.calc_dx_exp_x(self.upsilon_theta))
        filename = "cpp_gencode/Se2_Dx_exp_x.cpp"

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

        stream = cse_codegen(self.a.calc_dx_this_mul_exp_x_at_0(self.upsilon_theta))
        filename = "cpp_gencode/Se2_Dx_this_mul_exp_x_at_0.cpp"
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
        filename = "cpp_gencode/Se2_Dx_log_this.cpp"

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
