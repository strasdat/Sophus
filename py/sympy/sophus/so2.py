"""TODO(docstring)"""

import functools
import unittest

import sympy

from sophus.complex import Complex
from sophus.cse_codegen import cse_codegen
from sophus.matrix import vector2
from sophus.matrix import zero_vector2


class Rotation2:
    """2 dimensional group of orthogonal matrices with determinant 1"""

    def __init__(self, z):
        """internally represented by a unit complex number z"""
        self.z = z

    @staticmethod
    def exp(theta):
        """exponential map"""
        return Rotation2(Complex(sympy.cos(theta), sympy.sin(theta)))

    def log(self):
        """logarithmic map"""
        return sympy.atan2(self.z.imag, self.z.real)

    def calc_dx_log_this(self):
        """TODO(docstring)"""
        return sympy.diff(self.log(), self[0])

    def calc_dx_log_exp_x_times_this_at_0(self, x):
        """TODO(docstring)"""
        return sympy.diff((Rotation2.exp(x) * self).log(), x).limit(x, 0)

    def __repr__(self):
        return "Rotation2:" + repr(self.z)

    @staticmethod
    def hat(theta):
        """TODO(docstring)"""
        return sympy.Matrix([[0, -theta], [theta, 0]])

    def matrix(self):
        """returns matrix representation"""
        return sympy.Matrix([[self.z.real, -self.z.imag], [self.z.imag, self.z.real]])

    def __mul__(self, right):
        """left-multiplication
        either rotation concatenation or point-transform"""
        if isinstance(right, sympy.Matrix):
            assert right.shape == (2, 1), right.shape
            return self.matrix() * right
        if isinstance(right, Rotation2):
            return Rotation2(self.z * right.z)
        assert False, f"unsupported type: {type(right)}"

    def __getitem__(self, key):
        return self.z[key]

    @staticmethod
    def calc_dx_exp_x(x):
        """TODO(docstring)"""
        return sympy.Matrix(2, 1, lambda r, c: sympy.diff(Rotation2.exp(x)[r], x))

    @staticmethod
    def dx_exp_x_at_0():
        """TODO(docstring)"""
        return sympy.Matrix([0, 1])

    @staticmethod
    def calc_dx_exp_x_at_0(x):
        """TODO(docstring)"""
        return Rotation2.calc_dx_exp_x(x).limit(x, 0)

    def calc_dx_this_mul_exp_x_at_0(self, x):
        """TODO(docstring)"""
        return sympy.Matrix(
            2, 1, lambda r, c: sympy.diff((self * Rotation2.exp(x))[r], x)
        ).limit(x, 0)

    @staticmethod
    def dxi_x_matrix(_x, i):
        """TODO(docstring)"""
        if i == 0:
            return sympy.Matrix([[1, 0], [0, 1]])
        if i == 1:
            return sympy.Matrix([[0, -1], [1, 0]])
        assert False

    @staticmethod
    def calc_dxi_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(2, 2, lambda r, c: sympy.diff(x.matrix()[r, c], x[i]))

    @staticmethod
    def dx_exp_x_matrix(x):
        """TODO(docstring)"""
        mat_r = Rotation2.exp(x)
        dx_exp_x = Rotation2.calc_dx_exp_x(x)
        l = [dx_exp_x[j] * Rotation2.dxi_x_matrix(mat_r, j) for j in [0, 1]]
        return functools.reduce((lambda a, b: a + b), l)

    @staticmethod
    def calc_dx_exp_x_matrix(x):
        """TODO(docstring)"""
        return sympy.Matrix(
            2, 2, lambda r, c: sympy.diff(Rotation2.exp(x).matrix()[r, c], x)
        )

    @staticmethod
    def dx_exp_x_matrix_at_0():
        """TODO(docstring)"""
        return Rotation2.hat(1)

    @staticmethod
    def calc_dx_exp_x_matrix_at_0(x):
        """TODO(docstring)"""
        return sympy.Matrix(
            2, 2, lambda r, c: sympy.diff(Rotation2.exp(x).matrix()[r, c], x)
        ).limit(x, 0)


class TestSo2(unittest.TestCase):
    """TODO(docstring)"""

    def setUp(self):
        self.theta = sympy.symbols("theta", real=True)
        x, y = sympy.symbols("c[0] c[1]", real=True)
        p0, p1 = sympy.symbols("p0 p1", real=True)
        self.a = Rotation2(Complex(x, y))
        self.p = vector2(p0, p1)

    def test_exp_log(self):
        """TODO(docstring)"""
        for theta in [0.0, 0.5, 0.1]:
            w = Rotation2.exp(theta).log()
            self.assertAlmostEqual(theta, w)

    def test_matrix(self):
        """TODO(docstring)"""
        foo_rotation_bar = Rotation2.exp(self.theta)
        foo_transform_bar = foo_rotation_bar.matrix()
        point_bar = self.p
        p1_foo = foo_rotation_bar * point_bar
        p2_foo = foo_transform_bar * point_bar
        self.assertEqual(sympy.simplify(p1_foo - p2_foo), zero_vector2())

    def test_derivatives(self):
        """TODO(docstring)"""
        self.assertEqual(
            sympy.simplify(
                Rotation2.calc_dx_exp_x_at_0(self.theta) - Rotation2.dx_exp_x_at_0()
            ),
            sympy.Matrix.zeros(2, 1),
        )
        for i in [0, 1]:
            self.assertEqual(
                sympy.simplify(
                    Rotation2.calc_dxi_x_matrix(self.a, i)
                    - Rotation2.dxi_x_matrix(self.a, i)
                ),
                sympy.Matrix.zeros(2, 2),
            )

        self.assertEqual(
            sympy.simplify(
                Rotation2.dx_exp_x_matrix(self.theta)
                - Rotation2.calc_dx_exp_x_matrix(self.theta)
            ),
            sympy.Matrix.zeros(2, 2),
        )
        self.assertEqual(
            sympy.simplify(
                Rotation2.dx_exp_x_matrix_at_0()
                - Rotation2.calc_dx_exp_x_matrix_at_0(self.theta)
            ),
            sympy.Matrix.zeros(2, 2),
        )

    # pylint: disable=too-many-branches
    def test_codegen(self):
        """TODO(docstring)"""
        stream = cse_codegen(Rotation2.calc_dx_exp_x(self.theta))
        filename = "cpp_gencode/So2_Dx_exp_x.cpp"
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

        stream = cse_codegen(self.a.calc_dx_this_mul_exp_x_at_0(self.theta))
        filename = "cpp_gencode/So2_Dx_this_mul_exp_x_at_0.cpp"
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
        filename = "cpp_gencode/So2_Dx_log_this.cpp"

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

        stream = cse_codegen(self.a.calc_dx_log_exp_x_times_this_at_0(self.theta))
        filename = "cpp_gencode/So2_Dx_log_exp_x_times_this_at_0.cpp"

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
