"""TODO(docstring)"""
import functools
import unittest

import sympy

from sophus.cse_codegen import cse_codegen
from sophus.matrix import vector3
from sophus.matrix import zero_vector3
from sophus.matrix import squared_norm
from sophus.quaternion import Quaternion


class Rotation3:
    """3 dimensional group of orthogonal matrices with determinant 1"""

    def __init__(self, q):
        """internally represented by a unit quaternion q"""
        self.q = q

    @staticmethod
    def exp(v):
        """exponential map"""
        theta_sq = squared_norm(v)
        theta = sympy.sqrt(theta_sq)
        return Rotation3(
            Quaternion(sympy.cos(0.5 * theta), sympy.sin(0.5 * theta) / theta * v)
        )

    def log(self):
        """logarithmic map"""
        n = sympy.sqrt(squared_norm(self.q.vec))
        return 2 * sympy.atan(n / self.q.real) / n * self.q.vec

    def calc_dx_log_this(self):
        """TODO(docstring)"""
        return sympy.Matrix(3, 3, lambda r, c: sympy.diff(self.log()[r], self[c]))

    def calc_dx_log_exp_x_times_this_at_0(self, x):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                3, 3, lambda r, c: sympy.diff((Rotation3.exp(x) * self).log()[r], x[c])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .limit(x[2], 0)
        )

    def __repr__(self):
        return "Rotation3:" + repr(self.q)

    def inverse(self):
        """TODO(docstring)"""
        return Rotation3(self.q.conj())

    @staticmethod
    def hat(o):
        """TODO(docstring)"""
        return sympy.Matrix([[0, -o[2], o[1]], [o[2], 0, -o[0]], [-o[1], o[0], 0]])

    @staticmethod
    def vee(mat_omega):
        """vee-operator
        It takes the 3x3-matrix representation ``mat_omega`` and maps it to the
        corresponding vector representation of Lie algebra.

        This is the inverse of the hat-operator, see above.

        Precondition: ``mat_omega`` must have the following structure:

                       |  0 -c  b |
                       |  c  0 -a |
                       | -b  a  0 |
        """
        v = vector3(
            mat_omega.row(2).col(1), mat_omega.row(0).col(2), mat_omega.row(1).col(0)
        )
        return v

    def matrix(self):
        """returns matrix representation"""
        return sympy.Matrix(
            [
                [
                    1 - 2 * self.q.vec[1] ** 2 - 2 * self.q.vec[2] ** 2,
                    2 * self.q.vec[0] * self.q.vec[1] - 2 * self.q.vec[2] * self.q[3],
                    2 * self.q.vec[0] * self.q.vec[2] + 2 * self.q.vec[1] * self.q[3],
                ],
                [
                    2 * self.q.vec[0] * self.q.vec[1] + 2 * self.q.vec[2] * self.q[3],
                    1 - 2 * self.q.vec[0] ** 2 - 2 * self.q.vec[2] ** 2,
                    2 * self.q.vec[1] * self.q.vec[2] - 2 * self.q.vec[0] * self.q[3],
                ],
                [
                    2 * self.q.vec[0] * self.q.vec[2] - 2 * self.q.vec[1] * self.q[3],
                    2 * self.q.vec[1] * self.q.vec[2] + 2 * self.q.vec[0] * self.q[3],
                    1 - 2 * self.q.vec[0] ** 2 - 2 * self.q.vec[1] ** 2,
                ],
            ]
        )

    def __mul__(self, right):
        """left-multiplication
        either rotation concatenation or point-transform"""
        if isinstance(right, sympy.Matrix):
            assert right.shape == (3, 1), right.shape
            return (self.q * Quaternion(0, right) * self.q.conj()).vec
        if isinstance(right, Rotation3):
            return Rotation3(self.q * right.q)
        assert False, f"unsupported type: {type(right)}"

    def __getitem__(self, key):
        return self.q[key]

    @staticmethod
    def calc_dx_exp_x(x):
        """TODO(docstring)"""
        return sympy.Matrix(4, 3, lambda r, c: sympy.diff(Rotation3.exp(x)[r], x[c]))

    @staticmethod
    def dx_exp_x_at_0():
        """TODO(docstring)"""
        return sympy.Matrix(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]
        )

    @staticmethod
    def calc_dx_exp_x_at_0(x):
        """TODO(docstring)"""
        return Rotation3.calc_dx_exp_x(x).subs(x[0], 0).subs(x[1], 0).limit(x[2], 0)

    def calc_dx_this_mul_exp_x_at_0(self, x):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                4, 3, lambda r, c: sympy.diff((self * Rotation3.exp(x))[r], x[c])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .limit(x[2], 0)
        )

    def calc_dx_exp_x_mul_this_at_0(self, x):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                3, 4, lambda r, c: sympy.diff((self * Rotation3.exp(x))[c], x[r, 0])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .limit(x[2], 0)
        )

    @staticmethod
    def dxi_x_matrix(x, i):
        """TODO(docstring)"""
        if i == 0:
            return sympy.Matrix(
                [
                    [0, 2 * x[1], 2 * x[2]],
                    [2 * x[1], -4 * x[0], -2 * x[3]],
                    [2 * x[2], 2 * x[3], -4 * x[0]],
                ]
            )
        if i == 1:
            return sympy.Matrix(
                [
                    [-4 * x[1], 2 * x[0], 2 * x[3]],
                    [2 * x[0], 0, 2 * x[2]],
                    [-2 * x[3], 2 * x[2], -4 * x[1]],
                ]
            )
        if i == 2:
            return sympy.Matrix(
                [
                    [-4 * x[2], -2 * x[3], 2 * x[0]],
                    [2 * x[3], -4 * x[2], 2 * x[1]],
                    [2 * x[0], 2 * x[1], 0],
                ]
            )
        if i == 3:
            return sympy.Matrix(
                [
                    [0, -2 * x[2], 2 * x[1]],
                    [2 * x[2], 0, -2 * x[0]],
                    [-2 * x[1], 2 * x[0], 0],
                ]
            )
        assert False

    @staticmethod
    def calc_dxi_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(3, 3, lambda r, c: sympy.diff(x.matrix()[r, c], x[i]))

    @staticmethod
    def dxi_exp_x_matrix(x, i):
        """TODO(docstring)"""
        mat_r = Rotation3.exp(x)
        dx_exp_x = Rotation3.calc_dx_exp_x(x)
        l = [dx_exp_x[j, i] * Rotation3.dxi_x_matrix(mat_r, j) for j in [0, 1, 2, 3]]
        return functools.reduce((lambda a, b: a + b), l)

    @staticmethod
    def calc_dxi_exp_x_matrix(x, i):
        """TODO(docstring)"""
        return sympy.Matrix(
            3, 3, lambda r, c: sympy.diff(Rotation3.exp(x).matrix()[r, c], x[i])
        )

    @staticmethod
    def dxi_exp_x_matrix_at_0(i):
        """TODO(docstring)"""
        v = zero_vector3()
        v[i] = 1
        return Rotation3.hat(v)

    @staticmethod
    def calc_dxi_exp_x_matrix_at_0(x, i):
        """TODO(docstring)"""
        return (
            sympy.Matrix(
                3, 3, lambda r, c: sympy.diff(Rotation3.exp(x).matrix()[r, c], x[i])
            )
            .subs(x[0], 0)
            .subs(x[1], 0)
            .limit(x[2], 0)
        )


class TestSo3(unittest.TestCase):
    """TODO(docstring)"""

    def setUp(self):
        omega0, omega1, omega2 = sympy.symbols(
            "omega[0], omega[1], omega[2]", real=True
        )
        x, v0, v1, v2 = sympy.symbols("q.w() q.x() q.y() q.z()", real=True)
        p0, p1, p2 = sympy.symbols("p0 p1 p2", real=True)
        v = vector3(v0, v1, v2)
        self.omega = vector3(omega0, omega1, omega2)
        self.a = Rotation3(Quaternion(x, v))
        self.p = vector3(p0, p1, p2)

    def test_exp_log(self):
        """TODO(docstring)"""
        for o in [
            vector3(0.0, 1, 0.5),
            vector3(0.1, 0.1, 0.1),
            vector3(0.01, 0.2, 0.03),
        ]:
            w = Rotation3.exp(o).log()
            for i in range(0, 3):
                self.assertAlmostEqual(o[i], w[i])

    def test_matrix(self):
        """TODO(docstring)"""
        foo_rotation_bar = Rotation3.exp(self.omega)
        foo_transform_bar = foo_rotation_bar.matrix()
        point_bar = self.p
        p1_foo = foo_rotation_bar * point_bar
        p2_foo = foo_transform_bar * point_bar
        self.assertEqual(sympy.simplify(p1_foo - p2_foo), zero_vector3())

    def test_derivatives(self):
        """TODO(docstring)"""
        self.assertEqual(
            sympy.simplify(
                Rotation3.calc_dx_exp_x_at_0(self.omega) - Rotation3.dx_exp_x_at_0()
            ),
            sympy.Matrix.zeros(4, 3),
        )

        for i in [0, 1, 2, 3]:
            self.assertEqual(
                sympy.simplify(
                    Rotation3.calc_dxi_x_matrix(self.a, i)
                    - Rotation3.dxi_x_matrix(self.a, i)
                ),
                sympy.Matrix.zeros(3, 3),
            )
        for i in [0, 1, 2]:
            self.assertEqual(
                sympy.simplify(
                    Rotation3.dxi_exp_x_matrix(self.omega, i)
                    - Rotation3.calc_dxi_exp_x_matrix(self.omega, i)
                ),
                sympy.Matrix.zeros(3, 3),
            )
            self.assertEqual(
                sympy.simplify(
                    Rotation3.dxi_exp_x_matrix_at_0(i)
                    - Rotation3.calc_dxi_exp_x_matrix_at_0(self.omega, i)
                ),
                sympy.Matrix.zeros(3, 3),
            )

    # pylint: disable=too-many-branches
    def test_codegen(self):
        """TODO(docstring)"""
        stream = cse_codegen(Rotation3.calc_dx_exp_x(self.omega))
        filename = "cpp_gencode/So3_Dx_exp_x.cpp"
        # set to true to generate codegen files
        # pylint: disable=using-constant-test
        if False:
            with open(filename, encoding="utf-8") as file:
                for line in stream:
                    file.write(line)
        else:
            with open(filename, "r", encoding="utf-8") as file:
                file_lines = file.readlines()
                for i, line in enumerate(stream):
                    self.assertEqual(line, file_lines[i])

        stream = cse_codegen(self.a.calc_dx_this_mul_exp_x_at_0(self.omega))
        filename = "cpp_gencode/So3_Dx_this_mul_exp_x_at_0.cpp"
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
        filename = "cpp_gencode/So3_Dx_log_this.cpp"

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

        stream = cse_codegen(self.a.calc_dx_log_exp_x_times_this_at_0(self.omega))
        filename = "cpp_gencode/So3_Dx_log_exp_x_times_this_at_0.cpp"

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
