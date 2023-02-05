"""TODO(docstring)"""

import unittest

import sympy
from sympy import init_printing

from sophus.cse_codegen import cse_codegen
from sophus.matrix import proj
from sophus.matrix import vector2
from sophus.matrix import vector3
from sophus.matrix import vector4

init_printing(use_latex="mathjax")


class KannalaBrandtTransformCamera:
    """KannalaBrandt camera transform"""

    def __init__(self, focal_length, center, distortion):
        assert isinstance(focal_length, sympy.Matrix)
        assert focal_length.shape == (2, 1), focal_length.shape
        assert isinstance(center, sympy.Matrix)
        assert center.shape == (2, 1), center.shape
        assert isinstance(distortion, sympy.Matrix)
        assert distortion.shape == (4, 1), distortion.shape

        self.focal_length = focal_length
        self.center = center
        self.distortion = distortion

    # pylint: disable=too-many-locals
    def pixel_from_z1_plane(self, point_in_camera_z1_plane):
        """TODO(docstring)"""

        assert isinstance(point_in_camera_z1_plane, sympy.Matrix)
        assert point_in_camera_z1_plane.shape == (
            2,
            1,
        ), point_in_camera_z1_plane.shape

        k0, k1, k2, k3 = self.distortion
        x = point_in_camera_z1_plane[0]
        y = point_in_camera_z1_plane[1]

        radius_squared = x**2 + y**2

        radius = sympy.sqrt(radius_squared)
        radius_inverse = 1.0 / radius
        theta = sympy.atan(radius)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        r_distorted = theta * (
            1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8
        )
        scaling = r_distorted * radius_inverse

        return vector2(
            scaling * x * self.focal_length[0] + self.center[0],
            scaling * y * self.focal_length[1] + self.center[1],
        )

    def calc_dx_pixel_from_z1_plane_x(self, point_in_camera_z1_plane):
        """TODO(docstring)"""

        assert isinstance(point_in_camera_z1_plane, sympy.Matrix)
        assert point_in_camera_z1_plane.shape == (
            2,
            1,
        ), point_in_camera_z1_plane.shape

        return sympy.simplify(
            sympy.Matrix(
                2,
                2,
                lambda r, c: sympy.diff(
                    self.pixel_from_z1_plane(point_in_camera_z1_plane)[r],
                    point_in_camera_z1_plane[c],
                ),
            ),
        )

    def calc_dx_projpixel_from_z1_plane_x(self, point_in_camera):
        """TODO(docstring)"""

        assert isinstance(point_in_camera, sympy.Matrix)
        assert point_in_camera.shape == (3, 1), point_in_camera.shape

        return sympy.Matrix(
            2,
            3,
            lambda r, c: sympy.diff(
                self.pixel_from_z1_plane(proj(point_in_camera))[r], point_in_camera[c]
            ),
        )

    def calc_dx_proj_x(self, point_in_camera):
        """TODO(docstring)"""

        assert isinstance(point_in_camera, sympy.Matrix)
        assert point_in_camera.shape == (3, 1), point_in_camera.shape

        return sympy.Matrix(
            2, 3, lambda r, c: sympy.diff(proj(point_in_camera)[r], point_in_camera[c])
        )


class TestPinhole(unittest.TestCase):
    """TODO(docstring)"""

    def setUp(self):
        fx, fy = sympy.symbols("f[0], f[1]", real=True)
        cx, cy = sympy.symbols("c[0], c[1]", real=True)
        a, b = sympy.symbols("a b", real=True)
        x, y, z = sympy.symbols("x y z", real=True)
        self.focal_length = vector2(fx, fy)
        self.center = vector2(cx, cy)
        self.point_in_camera_z1_plane = vector2(a, b)
        self.point_in_camera = vector3(x, y, z)
        d0, d1, d2, d3 = sympy.symbols("k[0] k[1] k[2] k[3]")
        self.dist = vector4(d0, d1, d2, d3)

    def test_derivatives(self):
        """TODO(docstring)"""

        pinhole = KannalaBrandtTransformCamera(
            self.focal_length, self.center, self.dist
        )
        print("point_in_camera_z1_plane: ", self.point_in_camera_z1_plane)
        point_in_image = pinhole.pixel_from_z1_plane(self.point_in_camera_z1_plane)
        print("point_in_image: ", point_in_image)
        calc_dx_pixel_from_z1_plane_x = pinhole.calc_dx_pixel_from_z1_plane_x(
            self.point_in_camera_z1_plane
        )
        print("calc_dx_pixel_from_z1_plane_x: ", calc_dx_pixel_from_z1_plane_x)

    def test_codegen(self):
        """TODO(docstring)"""
        pinhole = KannalaBrandtTransformCamera(
            self.focal_length, self.center, self.dist
        )
        dx_pixel_from_z1_plane_x = pinhole.calc_dx_pixel_from_z1_plane_x(
            self.point_in_camera_z1_plane
        )
        stream = cse_codegen(dx_pixel_from_z1_plane_x)
        filename = "cpp_gencode/kannala_brandt/dx_pixel_from_z1_plane_x.cpp"
        # set to true to generate codegen files
        # pylint: disable=using-constant-test
        if True:
            with open(filename, "w", encoding="utf-8") as file:
                for line in stream:
                    file.write(line)
        else:
            with open(filename, "r", encoding="utf-8") as file:
                file_lines = file.readlines()
                for i, line in enumerate(stream):
                    self.assertEqual(line, file_lines[i])

        stream = cse_codegen(dx_pixel_from_z1_plane_x)
        filename = "cpp_gencode/kannala_brandt/dx_pixel_from_z1_plane_x.cpp"


if __name__ == "__main__":
    unittest.main()
