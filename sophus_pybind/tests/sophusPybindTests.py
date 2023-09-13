import unittest

import numpy as np
from sophus_pybind import interpolate, iterativeMean, SE3, SO3


class SophusPybindTest(unittest.TestCase):
    def test_sophus(self) -> None:
        rotvec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
        translational_part = np.array(
            [[2.13, 4.19, 8.5], [0.87, 1.89, 19.45], [10000, 0.01, 0], [1, 1, 1]]
        )
        quat_and_translation = np.array(
            [
                [1, 0, 0, 0, 2.13, 4.19, 8.5],
                [
                    0.8775825618903728,
                    0.479425538604203,
                    0,
                    0,
                    0.87,
                    -7.350739989577756,
                    17.2354392964228,
                ],
                [
                    0.8775825618903728,
                    0,
                    0.479425538604203,
                    0,
                    8414.709848078965,
                    0.01,
                    -4596.976941318602,
                ],
                [
                    0.647859344852457,
                    0.4398023303285789,
                    0.4398023303285789,
                    0.4398023303285789,
                    1,
                    1,
                    1,
                ],
            ]
        )

        ## exp
        so3vec = SO3.exp(rotvec)
        se3vec = SE3.exp(translational_part, rotvec)

        ## from quat and translation
        se3vec_from_quat = SE3.from_quat_and_translation(
            quat_and_translation[:, 0],
            quat_and_translation[:, 1:4],
            quat_and_translation[:, 4:7],
        )

        self.assertIsNone(np.testing.assert_equal(len(so3vec), 4))
        self.assertIsNone(np.testing.assert_equal(len(se3vec), 4))
        self.assertIsNone(np.testing.assert_equal(len(se3vec_from_quat), 4))

        for i in range(0, 4):
            ## log
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    so3vec[i].log()[0], se3vec[i].log()[0, 3:6]
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(so3vec[i].log()[0], rotvec[i])
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    se3vec[i].log()[0, 0:3], translational_part[i]
                )
            )

            ## matrix
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    so3vec[i].to_matrix(), se3vec[i].to_matrix3x4()[:, 0:3]
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    so3vec[i].to_matrix(), se3vec[i].to_matrix()[0:3, 0:3]
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    SO3.from_matrix(so3vec[i].to_matrix()).to_matrix(),
                    so3vec[i].to_matrix(),
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    SE3.from_matrix(se3vec[i].to_matrix()).to_matrix(),
                    se3vec[i].to_matrix(),
                )
            )

            ## rotation
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    so3vec[i].log(), se3vec[i].rotation().log()
                )
            )

            ## translation
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    se3vec[i].translation()[0], se3vec[i].to_matrix()[0:3, 3]
                )
            )

            ## quaternion
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    se3vec_from_quat[i].to_quat_and_translation()[0],
                    quat_and_translation[i],
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    se3vec_from_quat[i].to_quat_and_translation()[0, 0:4],
                    so3vec[i].to_quat()[0],
                )
            )

            ## inverse
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    se3vec[i].inverse().inverse().log(), se3vec[i].log()
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    so3vec[i].inverse().inverse().log(), so3vec[i].log()
                )
            )

            # operator
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    np.linalg.norm((so3vec[i] @ so3vec[i].inverse()).log()), 0
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    (se3vec[i] @ se3vec[i].inverse()).log(),
                    np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                )
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    (se3vec[i] @ se3vec[i].inverse()).log(),
                    np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                )
            )

        # set item
        so3vec[3] = SO3()
        se3vec[3] = SE3()
        self.assertIsNone(
            np.testing.assert_array_almost_equal(so3vec[3].log()[0], SO3().log()[0])
        )
        self.assertIsNone(
            np.testing.assert_array_almost_equal(se3vec[3].log()[0], SE3().log()[0])
        )

        # interpolate
        inter0 = interpolate(se3vec[0], se3vec[1], 0)
        inter1 = interpolate(se3vec[0], se3vec[1], 1)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(se3vec[0].log()[0], inter0.log()[0])
        )
        self.assertIsNone(
            np.testing.assert_array_almost_equal(se3vec[1].log()[0], inter1.log()[0])
        )

        # average
        inter_half = interpolate(se3vec[0], se3vec[1], 0.5)
        average01 = iterativeMean(se3vec[0:2])
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                average01.log()[0], inter_half.log()[0]
            )
        )
