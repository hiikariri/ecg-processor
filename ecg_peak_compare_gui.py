"""ECG R-Peak / HR comparison GUI: Pan-Tompkins vs Wavelet (1.py) on the same record.

Modeled on ecg_peak_gui.py, but runs BOTH detectors and overlays them:
  * PT  - pan_tompkins.detect_r_peaks      (adaptive double-threshold)
  * WT  - 1.py calculate_heart_rate detect (wavelet db4 + find_peaks)

HR is reported as 60 / mean(valid RR), RR kept in 0.3-2.0 s (1.py convention),
which is more robust than peak-count/duration.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import ECG.ecg_analysis as eca

DEFAULT_DATASET = Path("dataset_primer_1")
ECG_SUFFIX = "_ecg.csv"
TIME_COL = "Time (s)"
ECG_SOURCES = {"Raw": "ECG_Raw (V)", "Filtered": "ECG_Filtered (V)"}

# Records to drop from the batch comparison (e.g. pure-noise recordings).
SKIP_RECORDS = ("085_Rainhard",)

PT_STYLE = dict(color="tab:red", marker="x")
WT_STYLE = dict(color="tab:green", marker="o")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def list_records(dataset_dir):
    return [f.name[: -len(ECG_SUFFIX)]
            for f in sorted(dataset_dir.glob(f"*{ECG_SUFFIX}"))]


def short_label(record):
    """Trim the ``data_`` prefix and the ``_YYYYMMDD_HHMMSS`` timestamp."""
    s = re.sub(r"_\d{8}_\d{6}$", "", record)
    return s[len("data_"):] if s.startswith("data_") else s


def load_ecg(dataset_dir, record, source_col):
    df = pd.read_csv(dataset_dir / f"{record}{ECG_SUFFIX}")
    df.columns = df.columns.str.strip()
    time = np.array(df[TIME_COL].values, dtype=float)
    ecg = np.array(df[source_col].values, dtype=float)  # writable copy for pywt
    return time, ecg


def load_reference_hr(dataset_dir, record):
    """Return the ground-truth heart rate (bpm) from the metadata JSON, or NaN."""
    path = dataset_dir / f"{record}_metadata.json"
    if not path.exists():
        return np.nan
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return float(meta["ground_truth"]["heart_rate_bpm"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return np.nan


def sampling_rate(time):
    dt = np.median(np.diff(time))
    return 1.0 / dt if dt > 0 else np.nan


# Detection + HR now live in the ecg_analysis library:
#   eca.detect_r_peaks(ecg, fs, method)   method in {"pan_tompkins", "wavelet"}
#   eca.heart_rate_from_peaks(peaks, fs)
#   eca.instantaneous_heart_rate(peaks, fs, time=...)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class CompareGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG R-Peak & HR: Pan-Tompkins vs Wavelet (Hi-Me! 2.0)")
        self.resize(1280, 900)
        self.dataset_dir = DEFAULT_DATASET

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        # Dataset folder
        controls.addWidget(QtWidgets.QLabel("Dataset folder"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.dataset_dir))
        controls.addWidget(self.dataset_edit, 0, 1, 1, 3)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)
        controls.addWidget(browse_btn, 0, 4)

        # Record (with prev/next navigation)
        controls.addWidget(QtWidgets.QLabel("Record"), 1, 0)
        self.record_combo = QtWidgets.QComboBox()
        controls.addWidget(self.record_combo, 1, 1)
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(lambda: self.step_record(-1))
        controls.addWidget(self.prev_btn, 1, 2)
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.next_btn.clicked.connect(lambda: self.step_record(1))
        controls.addWidget(self.next_btn, 1, 3)
        self.record_combo.currentIndexChanged.connect(self._update_nav_buttons)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_records)
        controls.addWidget(refresh_btn, 1, 4)

        # ECG source column
        controls.addWidget(QtWidgets.QLabel("ECG source"), 1, 5)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(ECG_SOURCES.keys())
        self.source_combo.setCurrentText("Filtered")
        controls.addWidget(self.source_combo, 1, 6)

        # ECG view window (does not change detection, only the visible slice)
        controls.addWidget(QtWidgets.QLabel("View start (s)"), 2, 0)
        self.view_start_spin = QtWidgets.QDoubleSpinBox()
        self.view_start_spin.setRange(0.0, 1_000_000.0)
        self.view_start_spin.setDecimals(1)
        self.view_start_spin.setSingleStep(5.0)
        controls.addWidget(self.view_start_spin, 2, 1)
        controls.addWidget(QtWidgets.QLabel("View width (s)"), 2, 3)
        self.view_width_spin = QtWidgets.QDoubleSpinBox()
        self.view_width_spin.setRange(2.0, 1_000_000.0)
        self.view_width_spin.setDecimals(1)
        self.view_width_spin.setSingleStep(5.0)
        self.view_width_spin.setValue(15.0)
        controls.addWidget(self.view_width_spin, 2, 4)
        self.view_start_spin.valueChanged.connect(self.update_view)
        self.view_width_spin.valueChanged.connect(self.update_view)

        # Holds the most recent single-record plot context (for live view updates).
        self._ecg_ax = None
        self._ecg_tmax = 0.0

        # Buttons
        self.detect_btn = QtWidgets.QPushButton("Detect && Plot (both)")
        self.detect_btn.clicked.connect(self.detect_and_plot)
        controls.addWidget(self.detect_btn, 3, 0, 1, 2)

        self.batch_btn = QtWidgets.QPushButton("Compare all records")
        self.batch_btn.clicked.connect(self.compare_all_records)
        controls.addWidget(self.batch_btn, 3, 2, 1, 2)

        # Metrics
        self.metrics_label = QtWidgets.QLabel("Select a record and click Detect && Plot.")
        self.metrics_label.setStyleSheet("font-weight: bold;")
        self.metrics_label.setTextFormat(QtCore.Qt.RichText)
        root.addWidget(self.metrics_label)

        # Figure
        self.figure = Figure(figsize=(12, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas)

        self.refresh_records()

    # ----------------------------------------------------------------- #
    def browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(self.dataset_dir))
        if path:
            self.dataset_edit.setText(path)
            self.refresh_records()

    def refresh_records(self):
        self.dataset_dir = Path(self.dataset_edit.text().strip())
        self.record_combo.clear()
        if not self.dataset_dir.exists():
            self.metrics_label.setText(f"Folder not found: {self.dataset_dir}")
            return
        records = list_records(self.dataset_dir)
        self.record_combo.addItems(records)
        if records:
            self.metrics_label.setText(
                f"{len(records)} records loaded. Pick one and Detect && Plot.")
        else:
            self.metrics_label.setText(f"No *{ECG_SUFFIX} files in {self.dataset_dir}")
        self._update_nav_buttons()

    def step_record(self, delta):
        n = self.record_combo.count()
        if n == 0:
            return
        i = min(n - 1, max(0, self.record_combo.currentIndex() + delta))
        if i != self.record_combo.currentIndex():
            self.record_combo.setCurrentIndex(i)
            self.detect_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n = self.record_combo.count()
        i = self.record_combo.currentIndex()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)

    # ----------------------------------------------------------------- #
    def detect_and_plot(self):
        record = self.record_combo.currentText().strip()
        if not record:
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return

        source = self.source_combo.currentText()
        try:
            time, ecg = load_ecg(self.dataset_dir, record, ECG_SOURCES[source])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        fs = sampling_rate(time)
        pt_peaks = eca.detect_r_peaks(ecg, fs, "pan_tompkins")
        wt_peaks = eca.detect_r_peaks(ecg, fs, "wavelet")
        ref_hr = load_reference_hr(self.dataset_dir, record)

        pt_hr = eca.heart_rate_from_peaks(pt_peaks, fs)
        wt_hr = eca.heart_rate_from_peaks(wt_peaks, fs)

        def err_str(hr):
            return f"{abs(hr - ref_hr):.1f}" if np.isfinite(ref_hr) and np.isfinite(hr) else "-"

        self.metrics_label.setText(
            f"{record} | {source} | fs {fs:.0f} Hz &nbsp;&nbsp; "
            f"<span style='color:#d62728'>PT: {len(pt_peaks)} peaks, "
            f"HR {pt_hr:.1f} bpm (err {err_str(pt_hr)})</span> &nbsp;&nbsp; "
            f"<span style='color:#2ca02c'>WT: {len(wt_peaks)} peaks, "
            f"HR {wt_hr:.1f} bpm (err {err_str(wt_hr)})</span> &nbsp;&nbsp; "
            f"Ground-truth HR: {ref_hr:.0f} bpm"
        )

        # ---- Plot ----
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        ax1.plot(time, ecg, lw=0.7, color="tab:blue", label=f"ECG ({source})", zorder=1)
        if len(pt_peaks):
            # 'x' is an unfilled marker -> color via c, not edgecolors.
            ax1.scatter(time[pt_peaks], ecg[pt_peaks], s=55, c=PT_STYLE["color"],
                        linewidths=1.4, marker=PT_STYLE["marker"], zorder=3,
                        label="PT R-peaks")
        if len(wt_peaks):
            ax1.scatter(time[wt_peaks], ecg[wt_peaks], s=55, facecolors="none",
                        edgecolors=WT_STYLE["color"], linewidths=1.4,
                        marker=WT_STYLE["marker"], zorder=2, label="WT R-peaks")
        ax1.set_title(f"{record} - ECG with detected R-peaks (PT vs WT)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper right")

        # Beat-to-beat HR for both methods.
        pt_hr_t, pt_hr_y = eca.instantaneous_heart_rate(pt_peaks, fs, time=time)
        wt_hr_t, wt_hr_y = eca.instantaneous_heart_rate(wt_peaks, fs, time=time)
        if len(pt_hr_t):
            ax2.plot(pt_hr_t, pt_hr_y, "o-", ms=3, color=PT_STYLE["color"],
                     alpha=0.85, label="PT HR (beat-to-beat)")
        if len(wt_hr_t):
            ax2.plot(wt_hr_t, wt_hr_y, "s-", ms=3, color=WT_STYLE["color"],
                     alpha=0.85, label="WT HR (beat-to-beat)")
        if np.isfinite(pt_hr):
            ax2.axhline(pt_hr, ls=":", color=PT_STYLE["color"], alpha=0.7,
                        label=f"PT mean HR = {pt_hr:.1f}")
        if np.isfinite(wt_hr):
            ax2.axhline(wt_hr, ls=":", color=WT_STYLE["color"], alpha=0.7,
                        label=f"WT mean HR = {wt_hr:.1f}")
        if np.isfinite(ref_hr):
            ax2.axhline(ref_hr, ls="--", color="tab:blue",
                        label=f"Ground-truth HR = {ref_hr:.0f}")
        ax2.set_title("Heart rate: PT vs WT vs ground truth")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("HR (bpm)")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=8, ncol=2)

        self._ecg_ax = ax1
        self._ecg_tmax = float(time[-1])
        self._apply_view_window()
        self.canvas.draw_idle()

    def _apply_view_window(self):
        if self._ecg_ax is None:
            return
        start = max(0.0, self.view_start_spin.value())
        width = self.view_width_spin.value()
        end = min(start + width, self._ecg_tmax)
        if start >= end:
            start, end = 0.0, self._ecg_tmax
        self._ecg_ax.set_xlim(start, end)

    def update_view(self):
        if self._ecg_ax is None:
            return
        self._apply_view_window()
        self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def compare_all_records(self):
        records = [r for r in list_records(self.dataset_dir)
                   if short_label(r) not in SKIP_RECORDS]
        if not records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return

        source = self.source_combo.currentText()
        self._ecg_ax = None

        progress = QtWidgets.QProgressDialog(
            "Running both detectors on all records...", "Cancel", 0, len(records), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Batch comparison")
        progress.show()

        rows = []
        for i, record in enumerate(records):
            if progress.wasCanceled():
                break
            try:
                time, ecg = load_ecg(self.dataset_dir, record, ECG_SOURCES[source])
                ref_hr = load_reference_hr(self.dataset_dir, record)
            except Exception:
                progress.setValue(i + 1)
                continue

            fs = sampling_rate(time)
            pt_hr = eca.analyze(ecg, fs, "pan_tompkins").heart_rate
            wt_hr = eca.analyze(ecg, fs, "wavelet").heart_rate
            rows.append({"record": record, "ref_hr": ref_hr,
                         "pt_hr": pt_hr, "wt_hr": wt_hr})
            progress.setValue(i + 1)
        progress.close()

        res = pd.DataFrame(rows)
        valid = res.dropna(subset=["ref_hr", "pt_hr", "wt_hr"])
        if valid.empty:
            QtWidgets.QMessageBox.warning(self, "No results", "No valid comparisons produced.")
            return

        def stats(col):
            e = valid[col] - valid["ref_hr"]
            return (e.abs().mean(), np.sqrt((e ** 2).mean()), e.mean(),
                    np.corrcoef(valid[col], valid["ref_hr"])[0, 1])

        pt_mae, pt_rmse, pt_bias, pt_r = stats("pt_hr")
        wt_mae, wt_rmse, wt_bias, wt_r = stats("wt_hr")

        out = self.dataset_dir.parent / f"hr_compare_{source.lower()}.csv"
        res.to_csv(out, index=False)

        self.metrics_label.setText(
            f"Batch [{source}] n={len(valid)} &nbsp;&nbsp; "
            f"<span style='color:#d62728'>PT  MAE {pt_mae:.2f}  RMSE {pt_rmse:.2f}  "
            f"bias {pt_bias:+.2f}  r {pt_r:.3f}</span> &nbsp;&nbsp; "
            f"<span style='color:#2ca02c'>WT  MAE {wt_mae:.2f}  RMSE {wt_rmse:.2f}  "
            f"bias {wt_bias:+.2f}  r {wt_r:.3f}</span> &nbsp; (saved {out.name})"
        )

        # ---- Plot ----
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        x = valid["ref_hr"].values
        ax1.scatter(x, valid["pt_hr"].values, s=22, alpha=0.75,
                    color=PT_STYLE["color"], label=f"PT (MAE {pt_mae:.2f})")
        ax1.scatter(x, valid["wt_hr"].values, s=22, alpha=0.75,
                    color=WT_STYLE["color"], marker="s", label=f"WT (MAE {wt_mae:.2f})")
        lo = float(min(x.min(), valid[["pt_hr", "wt_hr"]].values.min()))
        hi = float(max(x.max(), valid[["pt_hr", "wt_hr"]].values.max()))
        ax1.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x")
        ax1.set_xlabel("Ground-truth HR (bpm)")
        ax1.set_ylabel("Estimated HR (bpm)")
        ax1.set_title(f"Estimated vs ground-truth HR - all records ({source})")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)

        idx = np.arange(len(valid))
        ax2.plot(idx, (valid["pt_hr"] - valid["ref_hr"]).abs().values, "o-", ms=3,
                 color=PT_STYLE["color"], alpha=0.8, label="PT abs err")
        ax2.plot(idx, (valid["wt_hr"] - valid["ref_hr"]).abs().values, "s-", ms=3,
                 color=WT_STYLE["color"], alpha=0.8, label="WT abs err")
        ax2.axhline(pt_mae, ls="--", color=PT_STYLE["color"], alpha=0.7)
        ax2.axhline(wt_mae, ls="--", color=WT_STYLE["color"], alpha=0.7)
        ax2.set_xticks(idx)
        ax2.set_xticklabels([short_label(r) for r in valid["record"]],
                            rotation=90, fontsize=6)
        ax2.set_xlabel("Record")
        ax2.set_ylabel("Abs HR error (bpm)")
        ax2.set_title("Per-record absolute HR error (PT vs WT)")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3, axis="y")

        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CompareGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
