"""TODO(docstring)"""

import unittest

import sympy

from sophus.cse_codegen import cse_codegen
from sophus.matrix import proj
from sophus.matrix import vector2
from sophus.matrix import vector3
from sophus.matrix import vector8


class BrownConradyCamera:
    """Brown Conrady camera transform"""

    def __init__(self, focal_length, center, distortion):
        assert isinstance(focal_length, sympy.Matrix)
        assert focal_length.shape == (2, 1), focal_length.shape
        assert isinstance(center, sympy.Matrix)
        assert center.shape == (2, 1), center.shape
        self.focal_length = focal_length
        self.center = center
        self.distortion = distortion

    # normalized_pixel is distorted coordinate on z1 plane
    def normalized_pixel_from_z1_plane(self, point_in_camera_z1_plane):
        """TODO(docstring)"""
        assert isinstance(point_in_camera_z1_plane, sympy.Matrix)
        assert point_in_camera_z1_plane.shape == (
            2,
            1,
        ), point_in_camera_z1_plane.shape
        x = point_in_camera_z1_plane[0]
        y = point_in_camera_z1_plane[1]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2.0 * x * y
        a2 = r2 + 2.0 * x * x
        a3 = r2 + 2.0 * y * y

        cdist = (
            1.0
            + self.distortion[0] * r2
            + self.distortion[1] * r4
            + self.distortion[4] * r6
        )
        icdist2 = 1.0 / (
            1.0
            + self.distortion[5] * r2
            + self.distortion[6] * r4
            + self.distortion[7] * r6
        )

        return vector2(
            x * cdist * icdist2 + self.distortion[2] * a1 + self.distortion[3] * a2,
            y * cdist * icdist2 + self.distortion[2] * a3 + self.distortion[3] * a1,
        )

    def pixel_from_z1_plane(self, point_in_camera_z1_plane):
        """TODO(docstring)"""
        assert isinstance(point_in_camera_z1_plane, sympy.Matrix)
        assert point_in_camera_z1_plane.shape == (
            2,
            1,
        ), point_in_camera_z1_plane.shape

        normalized_pixel = self.normalized_pixel_from_z1_plane(point_in_camera_z1_plane)
        x_prime = normalized_pixel[0]
        y_prime = normalized_pixel[1]

        return vector2(
            self.focal_length[0] * x_prime + self.center[0],
            self.focal_length[1] * y_prime + self.center[1],
        )

    def calc_dx_normalized_from_z1_plane_x(self, point_in_camera_z1_plane):
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
                    self.normalized_pixel_from_z1_plane(point_in_camera_z1_plane)[r],
                    point_in_camera_z1_plane[c],
                ),
            ),
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


class TestBrownConrady(unittest.TestCase):
    """TODO(docstring)"""

    def setUp(self):
        # pylint: disable=R0914
        fx, fy = sympy.symbols("f[0], f[1]", real=True)
        cx, cy = sympy.symbols("c[0], c[1]", real=True)
        a, b = sympy.symbols("a b", real=True)
        x, y, z = sympy.symbols("x y z", real=True)
        self.focal_length = vector2(fx, fy)
        self.center = vector2(cx, cy)
        self.point_in_camera_z1_plane = vector2(a, b)
        self.point_in_camera = vector3(x, y, z)
        d0, d1, d2, d3, d4, d5, d6, d7 = sympy.symbols(
            "d[0] d[1] d[2] d[3] d[4] d[5] d[6] d[7]"
        )
        self.dist = vector8(d0, d1, d2, d3, d4, d5, d6, d7)

    def test_derivatives(self):
        """TODO(docstring)"""
        pinhole = BrownConradyCamera(self.focal_length, self.center, self.dist)
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

    def test_codegen1(self):
        """TODO(docstring)"""
        pinhole = BrownConradyCamera(self.focal_length, self.center, self.dist)
        dx_pixel_from_z1_plane_x = pinhole.calc_dx_pixel_from_z1_plane_x(
            self.point_in_camera_z1_plane
        )
        stream = cse_codegen(dx_pixel_from_z1_plane_x)
        filename = "cpp_gencode/brown_conrady/dx_pixel_from_z1_plane_x.cpp"
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

    def test_codegen2(self):
        """TODO(docstring)"""
        pinhole = BrownConradyCamera(self.focal_length, self.center, self.dist)
        dx_normalized_from_z1_plane_x = pinhole.calc_dx_normalized_from_z1_plane_x(
            self.point_in_camera_z1_plane
        )
        stream = cse_codegen(dx_normalized_from_z1_plane_x)
        filename = "cpp_gencode/brown_conrady/dx_normalized_from_z1_plane_x.cpp"
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
