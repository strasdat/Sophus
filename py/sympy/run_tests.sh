#!/bin/bash

EXIT=0

python3 -m sophus.complex || EXIT=$?
python3 -m sophus.quaternion || EXIT=$?
python3 -m sophus.dual_quaternion || EXIT=$?
python3 -m sophus.so2 || EXIT=$?
python3 -m sophus.se2 || EXIT=$?
python3 -m sophus.so3 || EXIT=$?
python3 -m sophus.se3 || EXIT=$?

python3 -m sophus.inverse_depth
python3 -m sophus.affine_camera
python3 -m sophus.brown_conrady_camera
python3 -m sophus.kannala_brandt_camera

exit $EXIT
