import sys

import sympy

assert sys.version_info >= (3, 5)


def dot(left, right):
    assert isinstance(left, sympy.Matrix)
    assert isinstance(right, sympy.Matrix)

    sum = 0
    for c in range(0, left.cols):
        for r in range(0, left.rows):
            sum += left[r, c] * right[r, c]
    return sum


def squared_norm(m):
    assert isinstance(m, sympy.Matrix)
    return dot(m, m)


def Vector2(x, y):
    return sympy.Matrix([x, y])


def ZeroVector2():
    return Vector2(0, 0)


def Vector3(x, y, z):
    return sympy.Matrix([x, y, z])


def ZeroVector3():
    return Vector3(0, 0, 0)


def Vector4(a, b, c, d):
    return sympy.Matrix([a, b, c, d])


def ZeroVector4():
    return Vector4(0, 0, 0, 0)


def Vector5(a, b, c, d, e):
    return sympy.Matrix([a, b, c, d, e])


def ZeroVector5():
    return Vector5(0, 0, 0, 0, 0)


def Vector6(a, b, c, d, e, f):
    return sympy.Matrix([a, b, c, d, e, f])


def ZeroVector6():
    return Vector6(0, 0, 0, 0, 0, 0)


def proj(v):
    m, n = v.shape
    assert m > 1
    assert n == 1
    list = [v[i] / v[m - 1] for i in range(0, m - 1)]
    r = sympy.Matrix(m - 1, 1, list)
    return r


def unproj(v):
    m, n = v.shape
    assert m >= 1
    assert n == 1
    return v.col_join(sympy.Matrix.ones(1, 1))
