"""
Standalone agreement plot for a True HR vs Predicted HR dataset.
Mirrors the 'Evaluate all' visualisation in hr_test_gui.py:
  left  – agreement scatter with identity line and ±10%-of-true band
  right – signed-error histogram with bias line
"""

import numpy as np
import matplotlib.pyplot as plt

TOL_FRAC = 0.10  # ±10% of true HR band (ANSI/AAMI EC13, IEC 60601-2-27)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
true_hr = np.array([
    71, 66, 84, 88, 101, 85, 118, 89, 106, 90,
    82, 125, 80, 101, 95, 76, 99, 95, 87, 100,
    117, 85, 99, 96, 106, 79, 102, 118, 124, 111,
    94, 85, 92, 90, 99, 105, 78, 105, 87, 79,
    89, 75, 79, 88, 83, 95, 97, 109, 68, 99,
    99, 86, 74, 88, 84, 103, 90, 107, 101, 83,
    78, 90, 85, 79, 99, 109, 71, 79, 102, 104,
    80, 101, 86, 70, 75, 92, 94, 78, 90, 64,
    83, 86, 95, 95, 86, 102, 87, 92, 102, 103,
    97, 95, 86, 89, 76, 81, 67, 80, 87, 68,
    80, 88, 108, 92, 109, 80, 88, 88, 76,
], dtype=float)

pred_hr = np.array([
    72, 66, 91, 88, 103, 87, 118, 87, 106, 90,
    81, 125, 80, 101, 95, 76, 99, 95, 88, 101,
    117, 85, 100, 96, 107, 78, 102, 118, 124, 112,
    94, 84, 90, 95, 99, 105, 77, 105, 92, 79,
    89, 76, 79, 89, 83, 91, 96, 110, 69, 99,
    99, 87, 75, 91, 83, 106, 90, 107, 104, 81,
    79, 90, 85, 79, 101, 108, 69, 78, 101, 104,
    80, 101, 86, 70, 75, 92, 95, 78, 89, 62,
    75, 86, 94, 96, 85, 102, 89, 93, 102, 102,
    97, 95, 87, 88, 76, 81, 66, 80, 86, 69,
    73, 89, 111, 92, 109, 80, 88, 88, 76,
], dtype=float)

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
err  = pred_hr - true_hr
mae  = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err ** 2)))
bias = float(np.mean(err))
within = float(np.mean(np.abs(err) <= TOL_FRAC * true_hr) * 100)
n = len(true_hr)

print(f"n={n}  MAE {mae:.2f}  RMSE {rmse:.2f}  bias {bias:+.2f} bpm  "
      f"within ±{TOL_FRAC * 100:.0f}% of true: {within:.1f}%")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

# --- Agreement scatter ---
lo = float(min(true_hr.min(), pred_hr.min())) - 5
hi = float(max(true_hr.max(), pred_hr.max())) + 5

band_x = np.array([max(lo, 0.0), hi])  # HR is positive; ±10% widens with rate
ax1.fill_between(band_x,
                 band_x * (1 - TOL_FRAC),
                 band_x * (1 + TOL_FRAC),
                 color="0.8", alpha=0.4, label=f"±{TOL_FRAC * 100:.0f}% of true")
ax1.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="identity")
ax1.scatter(true_hr, pred_hr, s=30, alpha=0.75,
            edgecolors="0.3", lw=0.3, color="#1f77b4",
            label=f"n={n}")
ax1.set_xlim(lo, hi)
ax1.set_ylim(lo, hi)
ax1.set_xlabel("True HR (bpm)")
ax1.set_ylabel("Predicted HR (bpm)")
ax1.set_title("Heart Rate Agreement (Predicted vs True)")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left", fontsize=8)

# --- Error histogram ---
ax2.hist(err, bins=max(8, int(np.sqrt(n) * 2)),
         color="tab:blue", alpha=0.8, edgecolor="0.3")
ax2.axvline(0,    color="k",          lw=0.9)
ax2.axvline(bias, color="tab:purple", lw=1.5, label=f"bias {bias:+.2f}")
ax2.set_xlabel("Heart rate error  predicted − true (bpm)")
ax2.set_ylabel("Records")
ax2.set_title(f"Error distribution  (MAE {mae:.2f},  RMSE {rmse:.2f})")
ax2.grid(alpha=0.3, axis="y")
ax2.legend(loc="upper right", fontsize=8)

# summary in figure suptitle
fig.suptitle(
    f"MAE {mae:.2f}  ·  RMSE {rmse:.2f}  ·  bias {bias:+.2f} bpm  ·  "
    f"within ±{TOL_FRAC * 100:.0f}% of true: {within:.1f}%  (n={n})",
    fontsize=9, y=1.01,
)

plt.savefig("hr_agreement.png", dpi=150, bbox_inches="tight")
plt.show()
