"""
Step 4: check that K, R and t really are a decomposition of P.
These checks look at the arithmetic of step 3. They do not test whether the
camera model fits the data; that is what the reprojection error in step 5 is for.
"""
import numpy as np

from camera_utils import load_matrix

np.set_printoptions(precision=6, suppress=True)

P = load_matrix("projection_matrix.csv")
K = load_matrix("calibration_matrix_K.csv")
R = load_matrix("rotation_matrix_R.csv")
t = load_matrix("translation_vector_t.csv")
C = load_matrix("camera_centre_C.csv")

print("K is upper triangular:", np.allclose(np.tril(K, -1), 0))
print("K[2,2] =", K[2, 2])
print("det(R) =", np.linalg.det(R))
print("orthogonality error |R^T R - I| =", np.linalg.norm(R.T @ R - np.eye(3)))

# K[R|t] should give P back, but only up to a scale factor, so find the best
# scale s (least squares) before comparing.
P_rebuilt = K @ np.hstack([R, t.reshape(3, 1)])
s = np.sum(P * P_rebuilt) / np.sum(P_rebuilt * P_rebuilt)
print("scale between P and K[R|t]: s =", s)
print("reconstruction error |P - s K[R|t]| =", np.linalg.norm(P - s * P_rebuilt))

C_from_P = -np.linalg.solve(P[:, :3], P[:, 3])
print("camera centre from P (-M^-1 p4):", C_from_P)
print("camera centre from R and t (-R^T t):", C)

angle = np.degrees(np.arccos((np.trace(R) - 1) / 2))
print("rotation angle of R: %.2f degrees" % angle)
print("viewing direction (third row of R):", R[2])
print("skew K[0,1] = %.4f px, fy/fx = %.4f" % (K[0, 1], K[1, 1] / K[0, 0]))
