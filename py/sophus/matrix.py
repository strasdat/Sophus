import sympy
import sys

assert sys.version_info >= (3, 5)


class Matrix(sympy.Matrix):
    def __new__(cls, *args, **kwargs):
        return sympy.Matrix.__new__(cls, *args, **kwargs)

    def dot(self, right):
        sum = 0
        for c in range(0, self.cols):
            for r in range(0, self.rows):
                sum += self[r, c] * right[r, c]
        return sum

    def squared_norm(self):
        return self.dot(self)


class Vector2(Matrix):
    def __new__(cls, x, y):
        return Matrix.__new__(cls, [x, y])

    @staticmethod
    def zero():
        return Vector2(0, 0)


class Vector3(Matrix):
    def __new__(cls, x, y, z):
        return Matrix.__new__(cls, [x, y, z])

    def cross(self, right):
        return Vector3(self.y() * right.z() - self.z() * right.y(),
                       self.z() * right.x() - self.x() * right.z(),
                       self.x() * right.y() - self.y() * right.x())

    def x(self):
        return self[0, 0]

    def y(self):
        return self[1, 0]

    def z(self):
        return self[2, 0]

    @staticmethod
    def zero():
        return Vector3(0, 0, 0)


class Vector6(Matrix):
    def __new__(cls, a, b, c, d, e, f):
        return Matrix.__new__(cls, [a, b, c, d, e, f])

    @staticmethod
    def zero():
        return Vector6(0, 0, 0, 0, 0, 0)


def proj(v):
    m, n = v.shape
    assert m > 1
    assert n == 1
    l = [v[i] / v[m - 1] for i in range(0, m - 1)]
    r = sympy.Matrix(m - 1, 1,  l)
    return r


def unproj(v):
    m, n = v.shape
    assert m >= 1
    assert n == 1
    return v.col_join(sympy.Matrix.ones(1, 1))
