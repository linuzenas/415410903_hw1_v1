"""
Step 6 (extra): two checks that go beyond the three tasks.

(a) DLT with and without the normalization step.
(b) 5-fold hold-out: estimate P from 4/5 of the points and measure the error
    on the remaining 1/5. The errors in step 5 are measured on the same points
    that were used to estimate P, so they can only say how well the model fits
    those points. The hold-out error is a fairer test.
"""
import numpy as np

from camera_utils import load_points, estimate_projection_matrix, reprojection_errors

pts2d, pts3d = load_points()

# (a) normalized vs not normalized
for flag in (True, False):
    P, _ = estimate_projection_matrix(pts2d, pts3d, normalize=flag)
    err, _, _ = reprojection_errors(P, pts2d, pts3d)
    print("normalize=%-5s  average error %.6f px, median %.6f px, max %.6f px"
          % (flag, err.mean(), np.median(err), err.max()))

# (b) 5-fold hold-out with a fixed random seed
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(len(pts2d)), 5)
held_out = np.zeros(len(pts2d))
for fold in folds:
    train = np.setdiff1d(np.arange(len(pts2d)), fold)
    P, _ = estimate_projection_matrix(pts2d[train], pts3d[train])
    err, _, _ = reprojection_errors(P, pts2d[fold], pts3d[fold])
    held_out[fold] = err
print("5-fold hold-out: average %.6f px, median %.6f px, max %.6f px"
      % (held_out.mean(), np.median(held_out), held_out.max()))
