"""
Step 5: project the 3D points with P and measure the error (task 3).
error_i = distance in pixels between the given point (u_i, v_i) and the
projected point. Point numbers start at 1 and follow the order of the CSV.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt

from camera_utils import RESULTS, load_points, load_matrix, reprojection_errors

pts2d, pts3d = load_points()
P = load_matrix("projection_matrix.csv")
err, projected, w = reprojection_errors(P, pts2d, pts3d)

print("all points in front of the camera (w > 0):", bool(np.all(w > 0)))
print("average error: %.6f px" % err.mean())
print("median error:  %.6f px" % np.median(err))
print("maximum error: %.6f px" % err.max())
print("RMS error:     %.6f px" % np.sqrt(np.mean(err ** 2)))
print("90th / 99th percentile: %.4f / %.4f px" % tuple(np.percentile(err, [90, 99])))
print("points with error > 1 px: %d, > 2 px: %d" % ((err > 1).sum(), (err > 2).sum()))

order = np.argsort(-err)
print("five largest errors (point number, px):")
for i in order[:5]:
    print("  %3d  %.4f   observed (%.3f, %.3f)  projected (%.3f, %.3f)"
          % (i + 1, err[i], *pts2d[i], *projected[i]))
rest = np.delete(err, order[:2])
print("average error without the two largest points: %.6f px" % rest.mean())

table = pd.DataFrame({
    "point": np.arange(1, len(err) + 1),
    "u": pts2d[:, 0], "v": pts2d[:, 1],
    "X": pts3d[:, 0], "Y": pts3d[:, 1], "Z": pts3d[:, 2],
    "u_projected": projected[:, 0], "v_projected": projected[:, 1],
    "error_px": err,
})
RESULTS.mkdir(exist_ok=True)
table.to_csv(RESULTS / "reprojection_results.csv", index=False, float_format="%.10g")

# Figure 1: histogram of the errors
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(err, bins=30, color="#888888", edgecolor="black")
ax.axvline(err.mean(), color="black", linestyle="--", label="mean %.3f" % err.mean())
ax.axvline(np.median(err), color="black", linestyle=":", label="median %.3f" % np.median(err))
ax.set_xlabel("reprojection error (pixels)")
ax.set_ylabel("number of points")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "fig_error_histogram.png", dpi=200)
plt.close(fig)

# Figure 2: where in the image the errors are
fig, ax = plt.subplots(figsize=(6, 4.2))
sc = ax.scatter(pts2d[:, 0], pts2d[:, 1], c=err, s=14, cmap="viridis")
fig.colorbar(sc, label="error (pixels)")
for i in order[:2]:
    ax.annotate("point %d" % (i + 1), pts2d[i], textcoords="offset points",
                xytext=(6, 6), fontsize=8)
ax.invert_yaxis()                        # image coordinates: v grows downwards
ax.set_xlabel("u (pixels)")
ax.set_ylabel("v (pixels)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(RESULTS / "fig_error_map.png", dpi=200)
plt.close(fig)
