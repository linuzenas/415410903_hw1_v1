
import numpy as np
import pandas as pd

# Step 1: Load the original point correspondences.
data = pd.read_csv("cv_assignment_correspondences.csv")

points_2d = data[["u", "v"]].to_numpy(dtype=float)
points_3d = data[["X", "Y", "Z"]].to_numpy(dtype=float)

# Step 2: Load the projection matrix.
P = np.loadtxt("projection_matrix.csv", delimiter=",")

# Step 3: Convert 3D points into homogeneous coordinates.
ones = np.ones((len(points_3d), 1))
points_3d_h = np.hstack([points_3d, ones])

# Step 4: Project all 3D points into the image.
projected_h = (P @ points_3d_h.T).T

# Step 5: Divide by the third coordinate.
denominator = projected_h[:, 2]

if np.any(np.abs(denominator) < 1e-12):
    raise ValueError("A projected point has a near-zero denominator.")

predicted_2d = projected_h[:, :2] / denominator[:, None]

# Step 6: Calculate the error for each point.
differences = points_2d - predicted_2d
errors = np.linalg.norm(differences, axis=1)

# Step 7: Calculate average and other summary statistics.
average_error = np.mean(errors)
median_error = np.median(errors)
maximum_error = np.max(errors)

print("Number of points:", len(points_2d))
print(f"Average reprojection error: {average_error:.6f} pixels")
print(f"Median reprojection error:  {median_error:.6f} pixels")
print(f"Maximum reprojection error: {maximum_error:.6f} pixels")

# Step 8: Save actual and predicted coordinates.
results = pd.DataFrame({
    "u_actual": points_2d[:, 0],
    "v_actual": points_2d[:, 1],
    "u_predicted": predicted_2d[:, 0],
    "v_predicted": predicted_2d[:, 1],
    "error_pixels": errors
})

results.to_csv("reprojection_results.csv", index=False)

print("\nSaved point-by-point results to reprojection_results.csv")

# Step 9: Find the point with the largest reprojection error.
index = np.argmax(errors)

print("\nPoint with the largest error:")
print("Point number:", index + 1)
print("Actual 2D point:", points_2d[index])
print("Predicted 2D point:", predicted_2d[index])
print("Error in pixels:", errors[index])