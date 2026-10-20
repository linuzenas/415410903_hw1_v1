"""
Step 2: estimate the projection matrix P with the normalized DLT (task 1).
The method itself is explained in camera_utils.estimate_projection_matrix.
"""
import numpy as np

from camera_utils import load_points, save_matrix, estimate_projection_matrix

np.set_printoptions(precision=6, suppress=True)

pts2d, pts3d = load_points()
P, sv = estimate_projection_matrix(pts2d, pts3d, normalize=True)

print("number of correspondences:", len(pts2d))
print("design matrix shape: (%d, 12)" % (2 * len(pts2d)))
print("three smallest singular values of A (normalized coordinates):", sv[-3:])
print("projection matrix P:")
print(P)
print("norm of P[2,:3] =", np.linalg.norm(P[2, :3]))

save_matrix("projection_matrix.csv", P)
