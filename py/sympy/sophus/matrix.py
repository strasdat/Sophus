"""TODO(docstring)"""
import sys

import sympy

assert sys.version_info >= (3, 5)


def dot(left, right):
    """TODO(docstring)"""

    assert isinstance(left, sympy.Matrix)
    assert isinstance(right, sympy.Matrix)

    dot_sum = 0
    for c in range(0, left.cols):
        for r in range(0, left.rows):
            dot_sum += left[r, c] * right[r, c]
    return dot_sum


def squared_norm(m):
    """TODO(docstring)"""

    assert isinstance(m, sympy.Matrix)
    return dot(m, m)


def vector2(x, y):
    """TODO(docstring)"""

    return sympy.Matrix([x, y])


def zero_vector2():
    """TODO(docstring)"""

    return vector2(0, 0)


def vector3(x, y, z):
    """TODO(docstring)"""

    return sympy.Matrix([x, y, z])


def zero_vector3():
    """TODO(docstring)"""

    return vector3(0, 0, 0)


def vector4(a, b, c, d):
    """TODO(docstring)"""

    return sympy.Matrix([a, b, c, d])


def zero_vector4():
    """TODO(docstring)"""

    return vector4(0, 0, 0, 0)


def vector5(a, b, c, d, e):
    """TODO(docstring)"""

    return sympy.Matrix([a, b, c, d, e])


def zero_vector5():
    """TODO(docstring)"""

    return vector5(0, 0, 0, 0, 0)


def vector6(a, b, c, d, e, f):
    """TODO(docstring)"""

    return sympy.Matrix([a, b, c, d, e, f])


def zero_vector6():
    """TODO(docstring)"""

    return vector6(0, 0, 0, 0, 0, 0)


def vector7(a, b, c, d, e, f, g):
    """TODO(docstring)"""

    return sympy.Matrix([a, b, c, d, e, f, g])


def zero_vector7():
    """TODO(docstring)"""

    return vector7(0, 0, 0, 0, 0, 0, 0)


def vector8(a, b, c, d, e, f, g, h):
    """TODO(docstring)"""

    return sympy.Matrix([a, b, c, d, e, f, g, h])


def zero_vector8():
    """TODO(docstring)"""

    return vector8(0, 0, 0, 0, 0, 0, 0, 0)


def proj(v):
    """TODO(docstring)"""

    m, n = v.shape
    assert m > 1
    assert n == 1
    l = [v[i] / v[m - 1] for i in range(0, m - 1)]
    r = sympy.Matrix(m - 1, 1, l)
    return r


def unproj(v):
    """TODO(docstring)"""

    m, n = v.shape
    assert m >= 1
    assert n == 1
    return v.col_join(sympy.Matrix.ones(1, 1))
