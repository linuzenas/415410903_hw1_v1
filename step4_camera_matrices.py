
import numpy as np
from scipy.linalg import rq

# Step 1: Load the projection matrix from Step 3.
P = np.loadtxt("projection_matrix.csv", delimiter=",")

print("Projection matrix shape:", P.shape)

# Step 2: Split P into M and p4.
M = P[:, :3]
p4 = P[:, 3]

# Step 3: Use RQ decomposition to obtain K and R.
K, R = rq(M)

# Step 4: Adjust signs so the diagonal of K is positive.
signs = np.sign(np.diag(K))
signs[signs == 0] = 1

D = np.diag(signs)
K = K @ D
R = D @ R

# Step 5: Ensure the rotation matrix has determinant +1.
# If necessary, reverse the overall sign convention.
if np.linalg.det(R) < 0:
    K = -K
    R = -R

# Step 6: Normalize K so its bottom-right element is 1.
scale = K[2, 2]

if abs(scale) < 1e-12:
    raise ValueError("Cannot normalize K: K[2,2] is too close to zero.")

K = K / scale
p4 = p4 / scale

# Step 7: Calculate translation.
t = np.linalg.solve(K, p4)

print("\nCalibration matrix K:")
print(np.array2string(K, precision=6, suppress_small=True))

print("\nRotation matrix R:")
print(np.array2string(R, precision=6, suppress_small=True))

print("\nTranslation vector t:")
print(np.array2string(t, precision=6, suppress_small=True))

print("\nCheck K[2,2]:", K[2, 2])
print("Check determinant of R:", np.linalg.det(R))

# Save the matrices for the report.
np.savetxt("calibration_matrix_K.csv", K, delimiter=",", fmt="%.10f")
np.savetxt("rotation_matrix_R.csv", R, delimiter=",", fmt="%.10f")
np.savetxt("translation_vector_t.csv", t.reshape(1, 3),
           delimiter=",", fmt="%.10f")

print("\nSaved K, R, and t to CSV files.")
