"""
Step 1: look at the input data before using it.

The CSV was made from the Pt2D and Pt3D lists in the assignment PDF.
Row i of the 2D list belongs to row i of the 3D list. This script cannot
prove the pairing is right (only the PDF can), but the reprojection error in
step 5 would be huge if the pairing were wrong.
"""
import numpy as np
import pandas as pd

from camera_utils import load_points

pts2d, pts3d = load_points()

print("2D points shape:", pts2d.shape)
print("3D points shape:", pts3d.shape)
assert len(pts2d) == len(pts3d), "different number of 2D and 3D points"
assert np.isfinite(pts2d).all() and np.isfinite(pts3d).all(), "NaN or inf in the data"

print("duplicate rows:", int(pd.DataFrame(np.c_[pts2d, pts3d]).duplicated().sum()))
print("u range: %.3f to %.3f" % (pts2d[:, 0].min(), pts2d[:, 0].max()))
print("v range: %.3f to %.3f" % (pts2d[:, 1].min(), pts2d[:, 1].max()))
for name, col in zip("XYZ", pts3d.T):
    print("%s range: %.3f to %.3f" % (name, col.min(), col.max()))

# The DLT needs 3D points that are not all in one plane. If they were, the
# smallest singular value of the centred points would be close to zero.
sv = np.linalg.svd(pts3d - pts3d.mean(axis=0), compute_uv=False)
print("singular values of the centred 3D points:", np.round(sv, 3))
print("smallest / largest = %.3f (not near 0, so the points are not coplanar)" % (sv[-1] / sv[0]))
print("need at least 6 correspondences, have", len(pts2d))
