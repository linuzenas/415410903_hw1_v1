
import numpy as np
import pandas as pd


def normalize_2d(points):
    """Center 2D points and scale their average distance to sqrt(2)."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    distances = np.linalg.norm(centered, axis=1)
    mean_distance = np.mean(distances)

    if mean_distance < 1e-12:
        raise ValueError("2D points cannot all be identical.")

    scale = np.sqrt(2) / mean_distance

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    points_h = np.column_stack([points, np.ones(len(points))])
    normalized_h = (T @ points_h.T).T

    return normalized_h[:, :2], T


def normalize_3d(points):
    """Center 3D points and scale their average distance to sqrt(3)."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    distances = np.linalg.norm(centered, axis=1)
    mean_distance = np.mean(distances)

    if mean_distance < 1e-12:
        raise ValueError("3D points cannot all be identical.")

    scale = np.sqrt(3) / mean_distance

    T = np.array([
        [scale, 0, 0, -scale * centroid[0]],
        [0, scale, 0, -scale * centroid[1]],
        [0, 0, scale, -scale * centroid[2]],
        [0, 0, 0, 1]
    ])

    points_h = np.column_stack([points, np.ones(len(points))])
    normalized_h = (T @ points_h.T).T

    return normalized_h[:, :3], T


# Step 1: Load all corresponding points.
data = pd.read_csv("cv_assignment_correspondences.csv")

points_2d = data[["u", "v"]].to_numpy(dtype=float)
points_3d = data[["X", "Y", "Z"]].to_numpy(dtype=float)

assert len(points_2d) == len(points_3d)
assert np.isfinite(points_2d).all()
assert np.isfinite(points_3d).all()

# Step 2: Normalize the coordinates.
points_2d_norm, T2 = normalize_2d(points_2d)
points_3d_norm, T3 = normalize_3d(points_3d)

# Step 3: Build the DLT equation matrix A.
A_rows = []

for (u, v), (X, Y, Z) in zip(points_2d_norm, points_3d_norm):
    Xh = np.array([X, Y, Z, 1.0])

    row_u = np.concatenate([Xh, np.zeros(4), -u * Xh])
    row_v = np.concatenate([np.zeros(4), Xh, -v * Xh])

    A_rows.append(row_u)
    A_rows.append(row_v)

A = np.array(A_rows)

print("Number of correspondences:", len(points_2d))
print("Shape of matrix A:", A.shape)

# Step 4: Solve A p = 0 using SVD.
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# The last row of Vt gives the solution vector.
p = Vt[-1, :]

# Convert the 12 values into a 3 x 4 matrix.
P_normalized = p.reshape(3, 4)

# Step 5: Convert P back to the original coordinate systems.
P = np.linalg.inv(T2) @ P_normalized @ T3

# Step 6: Choose a convenient scale for displaying P.
scale = np.linalg.norm(P[2, :3])

if scale < 1e-12:
    raise ValueError("Could not normalize the projection matrix.")

P = P / scale

# The sign is arbitrary; use a consistent display convention.
if P[2, 3] < 0:
    P = -P

print("\nProjection matrix P:")
print(np.array2string(P, precision=6, suppress_small=True))

np.savetxt("projection_matrix.csv", P, delimiter=",", fmt="%.10f")
print("\nSaved projection matrix to projection_matrix.csv")
