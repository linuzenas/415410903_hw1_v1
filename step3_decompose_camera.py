"""
Step 3: split P into K, R and t (task 2).

P = [M | p4] = K [R | t] = [K R | K t], so
  * M = K R : K is upper triangular and R is a rotation, which is exactly an
    RQ decomposition of the left 3x3 block M;
  * t = K^-1 p4.
RQ is not unique: the signs of the columns of K (and rows of R) can be
flipped together. The signs are chosen so that the diagonal of K is positive.
K is divided by K[2,2] so that K[2,2] = 1; P is divided by the same number
so that P = K [R | t] still holds.

The camera centre in world coordinates is C = -R^T t.
"""
import numpy as np
from scipy.linalg import rq

from camera_utils import load_matrix, save_matrix

np.set_printoptions(precision=6, suppress=True)

P = load_matrix("projection_matrix.csv")
M, p4 = P[:, :3], P[:, 3]

K, R = rq(M)
signs = np.diag(np.sign(np.diag(K)))     # make the diagonal of K positive
K, R = K @ signs, signs @ R

if np.linalg.det(R) < 0:
    raise ValueError("det(R) is negative: P has the wrong overall sign")

scale = K[2, 2]
K = K / scale
t = np.linalg.solve(K, p4 / scale)
C = -R.T @ t

print("calibration matrix K:\n", K)
print("rotation matrix R:\n", R)
print("translation vector t:", t)
print("camera centre C = -R^T t:", C)

save_matrix("calibration_matrix_K.csv", K)
save_matrix("rotation_matrix_R.csv", R)
save_matrix("translation_vector_t.csv", t)
save_matrix("camera_centre_C.csv", C)
