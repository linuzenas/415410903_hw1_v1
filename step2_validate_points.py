"""
Computer Vision Assignment — Step 2: Load and validate corresponding points.

CSV columns:
u, v = 2D image coordinates (pixels)
X, Y, Z = corresponding 3D coordinates
"""

import numpy as np
import pandas as pd

# 1. Read the CSV file from the same folder as this Python script.
data = pd.read_csv("cv_assignment_correspondences.csv")

# 2. Separate the 2D and 3D coordinates into NumPy arrays.
points_2d = data[["u", "v"]].to_numpy(dtype=float)
points_3d = data[["X", "Y", "Z"]].to_numpy(dtype=float)

# 3. Print the basic shapes. Each row is one point.
print("2D points shape:", points_2d.shape)
print("3D points shape:", points_3d.shape)

# 4. Check that the number of corresponding points is the same.
assert points_2d.shape[0] == points_3d.shape[0], (
    "Error: the number of 2D points and 3D points does not match."
)

# 5. Check that every coordinate is a valid number.
assert np.isfinite(points_2d).all(), "Error: invalid value in 2D points."
assert np.isfinite(points_3d).all(), "Error: invalid value in 3D points."

# 6. Show a few rows so we can visually confirm the pairing.
print("\nFirst 5 corresponding rows:")
for i in range(min(5, len(data))):
    print(
        f"Point {i + 1}: "
        f"2D=({points_2d[i, 0]:.3f}, {points_2d[i, 1]:.3f})  "
        f"3D=({points_3d[i, 0]:.3f}, {points_3d[i, 1]:.3f}, "
        f"{points_3d[i, 2]:.3f})"
    )

print("\nValidation passed.")
print(f"Total corresponding points: {len(data)}")
