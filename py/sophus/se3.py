import sympy
import sys
import unittest
import sophus
import functools


class Se3:
    """ 3 dimensional group of rigid body transformations """

    def __init__(self, so3, t):
        """ internally represented by a unit quaternion q and a trnslation
            3-vector """
        self.so3 = so3
        self.t = t

    @staticmethod
    def exp(v):
        """ exponential map """
        upsilon = v[0:3, :]
        omega = sophus.Vector3(v[3], v[4], v[5])
        so3 = sophus.So3.exp(omega)
        Omega = sophus.So3.hat(omega)
        Omega_sq = Omega * Omega
        theta = sympy.sqrt(omega.squared_norm())
        V = (sophus.Matrix.eye(3) +
             (1 - sympy.cos(theta)) / (theta**2) * Omega +
             (theta - sympy.sin(theta)) / (theta**3) * Omega_sq)
        return Se3(so3, V * upsilon)

    def log(self):

        omega = self.so3.log()
        theta = sympy.sqrt(omega.squared_norm())
        Omega = sophus.So3.hat(omega)

        half_theta = 0.5 * theta

        V_inv = sophus.Matrix.eye(3) - 0.5 * Omega + (1 - theta * sympy.cos(
            half_theta) / (2 * sympy.sin(half_theta))) / (theta * theta) * (Omega * Omega)
        upsilon = V_inv * self.t
        return upsilon.col_join(omega)

    def __repr__(self):
        return "Se3: [" + repr(self.so3) + " " + repr(self.t)

    @staticmethod
    def hat(v):
        upsilon = sophus.Vector3(v[0], v[1], v[2])
        omega = sophus.Vector3(v[3], v[4], v[5])
        return sophus.So3.hat(omega).\
            row_join(upsilon).\
            col_join(sophus.Matrix.zeros(1, 4))

    def matrix(self):
        """ returns matrix representation """
        R = self.so3.matrix()
        return (R.row_join(self.t)).col_join(sophus.Matrix(1, 4, [0, 0, 0, 1]))

    def __mul__(self, right):
        """ left-multiplication
            either rotation concatenation or point-transform """
        if isinstance(right, sophus.Vector3):
            return self.so3 * right + self.t
        elif isinstance(right, So3):
            return So3(self.so3 * right.so3, self.t + self.so3 * right.t)
        assert False, "unsupported type: {0}".format(type(right))

    def __getitem__(self, key):
        """ We use the following convention [q0, q1, q2, q3, t0, t1, t2] """
        assert (key >= 0 and key < 7)
        if key < 4:
            return self.so3[key]
        else:
            return self.t[key - 4]

    @staticmethod
    def Dx_exp_x(x):
        omega = sophus.Vector3(x[3], x[4], x[5])
        o0 = omega[0]
        o1 = omega[1]
        o2 = omega[2]
        u0 = x[0]
        u1 = x[1]
        u2 = x[2]
        n_sq = omega.squared_norm()
        n = sympy.sqrt(n_sq)
        n_star = n_sq**sympy.Rational(-3, 2)
        sin_n = sympy.sin(n)
        cos_n = sympy.cos(n)
        foo = (n - sin_n) * n_star
        daz = (-cos_n + 1) / n_sq

        Du_exp_x_q = sophus.Matrix.zeros(3, 4)
        Du_exp_x_t = sophus.Matrix([[
            (-o1**2 - o2**2) * foo + 1,
            o0 * o1 * foo + o2 * daz,
            o0 * o2 * foo - o1 * daz,
        ], [
            o1 * o0 * foo - o2 * daz,
            (-o0**2 - o2**2) * foo + 1,
            o1 * o2 * foo + o0 * daz,

        ], [
            o2 * o0 * foo + o1 * daz,
            o2 * o1 * foo - o0 * daz,
            (-o0**2 - o1**2) * foo + 1,
        ]])
        Du_exp_x = Du_exp_x_q.row_join(Du_exp_x_t)
        Do_exp_x_q = sophus.So3.Dx_exp_x(omega)

        cos_n_by_n = cos_n / n

        a0 = (-o0*cos_n_by_n + o0/n) * n_star
        a1 = (-o1*cos_n_by_n + o1/n) * n_star
        a2 = (-o2*cos_n_by_n + o2/n) * n_star
        b = (n - sin_n) * n_sq**sympy.Rational(-5, 2)
        c = (-cos_n + 1)/n_sq**2
        d = sin_n * n_star
        e = (n - sin_n) * n_star
        f = (-cos_n + 1)/n_sq
        o00 = o0**2
        o11 = o1**2
        o22 = o2**2
        oo01 = (-o00 - o11)
        oo02 = (-o00 - o22)
        oo12 = (-o11 - o22)
        o01 = o0*o1
        o02 = o0*o2
        o12 = o1*o2
        m3_o001 = -3*o00*o1
        m3_o002 = -3*o00*o2
        m3_o011 = -3*o0*o11
        m3_o012 = -3*o01*o2
        m3_o022 = -3*o0*o22
        m3_o112 = -3*o11*o2
        m3_o122 = -3*o1*o22
        ab0 = o01*a0 + m3_o001*b
        ab1 = o02*a0 + m3_o002*b
        ab2 = o12*a0 + m3_o012*b
        ab3 = o01*a1 + m3_o011*b
        ab4 = o02*a1 + m3_o012*b
        ab5 = o12*a1 + m3_o112*b
        ab6 = o01*a2 + m3_o012*b
        ab7 = o02*a2 + m3_o022*b
        ab8 = o12*a2 + m3_o122*b

        Do_exp_x_t = sophus.Matrix([[
            u0*(-3*o0*oo12*b + oo12*a0) +
            u1*(ab0 + 2*o02*c - o02*d + o1*e) +
            u2*(ab1 - 2*o01*c + o01*d + o2*e),
            #
            u0*(ab0 - 2*o02*c + o02*d + o1*e) +
            u1*(-3*o0*oo02*b - 2*o0*e + oo02*a0) +
            u2*(ab2 + 2*o00*c - o00*d - f),
            #
            u0*(ab1 + 2*o01*c - o01*d + o2*e) +
            u1*(ab2 - 2*o00*c + o00*d + f) +
            u2*(-3*o0*oo01*b - 2*o0*e + oo01*a0)
        ], [
            u0*(-3*o1*oo12*b - 2*o1*e + oo12*a1) +
            u1*(ab3 + 2*o12*c - o12*d + o0*e) +
            u2*(ab4 - 2*o11*c + o11*d + f),
            #
            u0*(ab3 - 2*o12*c + o12*d + o0*e) +
            u1*(-3*o1*oo02*b + oo02*a1) +
            u2*(ab5 + 2*o01*c - o01*d + o2*e),
            #
            u0*(ab4 + 2*o11*c - o11*d - f) +
            u1*(ab5 - 2*o01*c + o01*d + o2*e) +
            u2*(-3*o1*oo01*b - 2*o1*e + oo01*a1)
        ], [
            u0*(-3*o2*oo12*b - 2*o2*e + oo12*a2) +
            u1*(ab6 + 2*o22*c - o22*d - f) +
            u2*(ab7 - 2*o12*c + o12*d + o0*e),
            #
            u0*(ab6 - 2*o22*c + o22*d + f) +
            u1*(-3*o2*oo02*b - 2*o2*e + oo02*a2) +
            u2*(ab8 + 2*o02*c - o02*d + o1*e),
            #
            u0*(ab7 + 2*o12*c - o12*d + o0*e) +
            u1*(ab8 - 2*o02*c + o02*d + o1*e) +
            u2*(-3*o2*oo01*b + oo01*a2)
        ]])
        Do_exp_x = Do_exp_x_q.row_join(Do_exp_x_t)
        return Du_exp_x.col_join(Do_exp_x)

    @staticmethod
    def calc_Dx_exp_x(x):
        return sophus.Matrix(6, 7, lambda r, c:
                             sympy.diff(Se3.exp(x)[c], x[r, 0]))

    @staticmethod
    def Dx_exp_x_at_0():
        return sophus.Matrix([[0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1],
                              [0.5, 0, 0, 0, 0, 0, 0],
                              [0, 0.5, 0, 0, 0, 0, 0],
                              [0, 0, 0.5, 0, 0, 0, 0]])

    @staticmethod
    def calc_Dx_exp_x_at_0(x):
        return Se3.Dx_exp_x(x).subs(x[0], 0).subs(x[1], 0).subs(x[2], 0).\
                   subs(x[3], 0).subs(x[4], 0).limit(x[5], 0)

    @staticmethod
    def Dxi_x_matrix(x, i):
        if i < 4:
            return sophus.So3.Dxi_x_matrix(x, i).\
                          row_join(sophus.Matrix.zeros(3, 1)).\
                          col_join(sophus.Matrix.zeros(1, 4))
        M = sophus.Matrix.zeros(4, 4)
        M[i-4, 3] = 1
        return M

    @staticmethod
    def calc_Dxi_x_matrix(x, i):
        return sophus.Matrix(4, 4, lambda r, c:
                             sympy.diff(x.matrix()[r, c], x[i]))

    @staticmethod
    def Dxi_exp_x_matrix(x, i):
        T = Se3.exp(x)
        Dx_exp_x = Se3.Dx_exp_x(x)
        l = [Se3.Dxi_x_matrix(T, j) * Dx_exp_x[i, j] for j in range(0, 7)]
        return functools.reduce((lambda a, b: a + b), l)

    @staticmethod
    def calc_Dxi_exp_x_matrix(x, i):
        return sophus.Matrix(4, 4, lambda r, c:
                             sympy.diff(Se3.exp(x).matrix()[r, c], x[i]))

    @staticmethod
    def Dxi_exp_x_matrix_at_0(i):
        v = sophus.Vector6.zero()
        v[i] = 1
        return Se3.hat(v)

    @staticmethod
    def calc_Dxi_exp_x_matrix_at_0(x, i):
        return sophus.Matrix(4, 4, lambda r, c:
                             sympy.diff(Se3.exp(x).matrix()[r, c], x[i])
                             ).subs(x[0], 0).subs(x[1], 0).subs(x[2], 0).\
                               subs(x[3], 0).subs(x[4], 0).limit(x[5], 0)


class TestSe3(unittest.TestCase):
    def setUp(self):
        u0, u1, u2, o0, o1, o2 = sympy.symbols(
            'u0, u1, u2, o0, o1, o2', real=True)
        x, v0, v1, v2 = sympy.symbols('x v0 v1 v2', real=True)
        p0, p1, p2 = sympy.symbols('p0 p1 p2', real=True)
        t0, t1, t2 = sympy.symbols('t0 t1 t2', real=True)
        v = sophus.Vector3(v0, v1, v2)
        self.o = sophus.Vector6(u0, u1, u2, o0, o1, o2)
        self.t = sophus.Vector3(t0, t1, t2)
        self.a = Se3(sophus.So3(sophus.Quaternion(x, v)), self.t)
        self.p = sophus.Vector3(p0, p1, p2)

    def test_exp_log(self):
        for v in [sophus.Vector6(0., 1, 0.5, 2., 1, 0.5),
                  sophus.Vector6(0.1, 0.1, 0.1, 0., 1, 0.5),
                  sophus.Vector6(0.01, 0.2, 0.03, 0.01, 0.2, 0.03)]:
            w = Se3.exp(v).log()
            for i in range(0, 3):
                self.assertAlmostEqual(v[i], w[i])

    def test_matrix(self):
        T_foo_bar = Se3.exp(self.o)
        Tmat_foo_bar = T_foo_bar.matrix()
        point_bar = self.p
        p1_foo = T_foo_bar * point_bar
        p2_foo = sophus.proj(Tmat_foo_bar * sophus.unproj(point_bar))
        self.assertEqual(sympy.simplify(p1_foo - p2_foo),
                         sophus.Vector3.zero())

    def test_derivatives(self):
        self.assertEqual(sympy.simplify(Se3.calc_Dx_exp_x(self.o) -
                                        Se3.Dx_exp_x(self.o)),
                         sophus.Matrix.zeros(6, 7))
        self.assertEqual(sympy.simplify(Se3.calc_Dx_exp_x_at_0(self.o) -
                                        Se3.Dx_exp_x_at_0()),
                         sophus.Matrix.zeros(6, 7))
        for i in range(0, 7):
            self.assertEqual(sympy.simplify(Se3.calc_Dxi_x_matrix(self.a, i) -
                                            Se3.Dxi_x_matrix(self.a, i)),
                             sophus.Matrix.zeros(4, 4))
        for i in range(0, 6):
            self.assertEqual(sympy.simplify(
                Se3.Dxi_exp_x_matrix(self.o, i) -
                Se3.calc_Dxi_exp_x_matrix(self.o, i)),
                sophus.Matrix.zeros(4, 4))
            self.assertEqual(sympy.simplify(
                Se3.Dxi_exp_x_matrix_at_0(i) -
                Se3.calc_Dxi_exp_x_matrix_at_0(self.o, i)),
                sophus.Matrix.zeros(4, 4))


if __name__ == '__main__':
    unittest.main()
