"""

The camera model is   x ~ P X   where x = (u, v, 1) is a pixel in homogeneous
form, X = (X, Y, Z, 1) is a 3D point, and P is a 3x4 matrix defined only up
to a scale factor.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # folder that contains the scripts
RESULTS = HERE / "results"                      # all output files go here
CSV_FILE = HERE / "cv_assignment_correspondences.csv"


def load_points():
    """Read the CSV and return the pixel points (N x 2) and 3D points (N x 3)."""
    table = pd.read_csv(CSV_FILE)
    pts2d = table[["u", "v"]].to_numpy(dtype=float)
    pts3d = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    return pts2d, pts3d


def to_homogeneous(points):
    """Append a column of ones: (N x k) -> (N x k+1)."""
    return np.hstack([points, np.ones((len(points), 1))])


def normalization_matrix(points, target_distance):
    """
    Similarity transform T that moves the centroid of the points to the origin
    and scales them so the average distance from the origin is target_distance
    (sqrt(2) for 2D points, sqrt(3) for 3D points).
    Returns a (k+1) x (k+1) matrix that acts on homogeneous coordinates.
    """
    k = points.shape[1]
    centroid = points.mean(axis=0)
    mean_dist = np.linalg.norm(points - centroid, axis=1).mean()
    s = target_distance / mean_dist
    T = np.eye(k + 1)
    T[:k, :k] *= s
    T[:k, k] = -s * centroid
    return T


def estimate_projection_matrix(pts2d, pts3d, normalize=True):
    """
    Estimate P with the Direct Linear Transform (DLT).

    Every correspondence gives two equations. From u = (p1.X)/(p3.X) and
    v = (p2.X)/(p3.X), where p1, p2, p3 are the rows of P:
        p1.X - u*(p3.X) = 0
        p2.X - v*(p3.X) = 0
    These are linear in the 12 entries of P, so with N points we get a
    2N x 12 matrix A and we look for p with A p = 0 and |p| = 1. The answer
    is the right singular vector of A that belongs to the smallest singular
    value.

    normalize=True shifts and scales both point sets first (Hartley
    normalization) so that A is better conditioned. P is then converted back
    to the original coordinates at the end.

    Returns P (scaled so that the norm of its first three entries in the
    third row is 1, and with the sign chosen so that the points lie in
    front of the camera) and the singular values of A.
    """
    n = len(pts2d)
    if normalize:
        T2 = normalization_matrix(pts2d, np.sqrt(2))
        T3 = normalization_matrix(pts3d, np.sqrt(3))
    else:
        T2, T3 = np.eye(3), np.eye(4)

    x = (to_homogeneous(pts2d) @ T2.T)[:, :2]       # normalized pixels
    X = to_homogeneous(pts3d) @ T3.T                # normalized 3D points (homogeneous)

    A = np.zeros((2 * n, 12))
    for i in range(n):
        u, v = x[i]
        A[2 * i, 0:4] = X[i]
        A[2 * i, 8:12] = -u * X[i]
        A[2 * i + 1, 4:8] = X[i]
        A[2 * i + 1, 8:12] = -v * X[i]

    _, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    P_norm = Vt[-1].reshape(3, 4)                   # smallest singular value

    P = np.linalg.inv(T2) @ P_norm @ T3             # back to the original coordinates

    # P is only known up to scale (and sign). Fix both.
    P = P / np.linalg.norm(P[2, :3])
    depth = to_homogeneous(pts3d) @ P[2]
    if np.median(depth) < 0:
        P = -P
    return P, singular_values


def project_points(P, pts3d):
    """
    Project 3D points with P. Returns the pixel coordinates (N x 2) and the
    homogeneous denominator w (N,). Points with w close to 0 cannot be
    projected, so that case raises an error instead of dividing by ~0.
    """
    proj = to_homogeneous(pts3d) @ P.T
    w = proj[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("a point has w close to zero, it cannot be projected")
    return proj[:, :2] / w[:, None], w


def reprojection_errors(P, pts2d, pts3d):
    """Euclidean distance in pixels between observed and projected points."""
    projected, w = project_points(P, pts3d)
    return np.linalg.norm(projected - pts2d, axis=1), projected, w


def save_matrix(name, M):
    """Save with full precision so that later scripts read back exactly what was computed."""
    RESULTS.mkdir(exist_ok=True)
    np.savetxt(RESULTS / name, np.atleast_2d(M), delimiter=",", fmt="%.17g")


def load_matrix(name):
    return np.loadtxt(RESULTS / name, delimiter=",")
