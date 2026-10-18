
import numpy as np

P = np.loadtxt("projection_matrix.csv", delimiter=",")
K = np.loadtxt("calibration_matrix_K.csv", delimiter=",")
R = np.loadtxt("rotation_matrix_R.csv", delimiter=",")
t = np.loadtxt("translation_vector_t.csv", delimiter=",")

# Check whether R transpose times R is close to identity.
identity_check = R.T @ R

# Reconstruct the projection matrix.
P_reconstructed = K @ np.column_stack((R, t))

# P and the reconstructed matrix may differ by an overall scale.
scale = np.sum(P * P_reconstructed) / np.sum(P_reconstructed**2)
P_reconstructed_scaled = scale * P_reconstructed

print("R transpose times R:")
print(identity_check)

print("\nRotation orthogonality error:")
print(np.linalg.norm(identity_check - np.eye(3)))

print("\nProjection matrix reconstruction error:")
print(np.linalg.norm(P - P_reconstructed_scaled))
