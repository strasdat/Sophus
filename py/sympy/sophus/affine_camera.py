"""TODO(docstring)"""

import unittest

import sympy

from sophus.cse_codegen import cse_codegen
from sophus.matrix import proj
from sophus.matrix import vector2
from sophus.matrix import vector3


class AffineCamera:
    """Affine camera transform"""

    def __init__(self, focal_length, center):
        assert isinstance(focal_length, sympy.Matrix)
        assert focal_length.shape == (2, 1), focal_length.shape
        assert isinstance(center, sympy.Matrix)
        assert center.shape == (2, 1), center.shape
        self.focal_length = focal_length
        self.center = center

    def pixel_from_z1_plane(self, point_in_camera_z1_plane):
        """Map point from z1-plane to"""
        assert isinstance(point_in_camera_z1_plane, sympy.Matrix)
        assert point_in_camera_z1_plane.shape == (
            2,
            1,
        ), point_in_camera_z1_plane.shape

        return vector2(
            self.focal_length[0] * point_in_camera_z1_plane[0] + self.center[0],
            self.focal_length[1] * point_in_camera_z1_plane[1] + self.center[1],
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
        """TODO(docstring)"""
        fx, fy = sympy.symbols("f[0], f[1]", real=True)
        cx, cy = sympy.symbols("c[0], c[1]", real=True)
        a, b = sympy.symbols("a b", real=True)
        x, y, z = sympy.symbols("x y z", real=True)
        self.focal_length = vector2(fx, fy)
        self.center = vector2(cx, cy)
        self.point_in_camera_z1_plane = vector2(a, b)
        self.point_in_camera = vector3(x, y, z)

    def test_derivatives(self):
        """TODO(docstring)"""
        pinhole = AffineCamera(self.focal_length, self.center)
        print("point_in_camera_z1_plane: ", self.point_in_camera_z1_plane)
        point_in_image = pinhole.pixel_from_z1_plane(self.point_in_camera_z1_plane)
        print("point_in_image: ", point_in_image)
        dx_pixel_from_z1_plane_x = pinhole.calc_dx_pixel_from_z1_plane_x(
            self.point_in_camera_z1_plane
        )
        print("Dx_pixel_from_z1_plane_x: ", dx_pixel_from_z1_plane_x)
        dx_projpixel_from_z1_plane_x = pinhole.calc_dx_projpixel_from_z1_plane_x(
            self.point_in_camera
        )
        print("Dx_projpixel_from_z1_plane_x: ", dx_projpixel_from_z1_plane_x)
        calc_dx_proj_x = pinhole.calc_dx_proj_x(self.point_in_camera)
        print("calc_dx_proj_x: ", calc_dx_proj_x)

    def test_codegen(self):
        """TODO(docstring)"""
        pinhole = AffineCamera(self.focal_length, self.center)
        dx_pixel_from_z1_plane_x = pinhole.calc_dx_pixel_from_z1_plane_x(
            self.point_in_camera_z1_plane
        )
        stream = cse_codegen(dx_pixel_from_z1_plane_x)
        filename = "cpp_gencode/affine/dx_pixel_from_z1_plane_x.cpp"
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

        stream = cse_codegen(dx_pixel_from_z1_plane_x)
        filename = "cpp_gencode/affine/dx_pixel_from_z1_plane_x.cpp"


if __name__ == "__main__":
    unittest.main()
