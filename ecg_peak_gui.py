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

from pan_tompkins import detect_r_peaks

DEFAULT_DATASET = Path("dataset_primer_1")
ECG_SUFFIX = "_ecg.csv"
TIME_COL = "Time (s)"
ECG_SOURCES = {"Raw": "ECG_Raw (V)", "Filtered": "ECG_Filtered (V)"}


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
    time = df[TIME_COL].values
    ecg = df[source_col].values
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


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class PeakGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG R-Peak Detection & HR Comparison (Hi-Me! 2.0)")
        self.resize(1280, 880)
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
        self.detect_btn = QtWidgets.QPushButton("Detect && Plot")
        self.detect_btn.clicked.connect(self.detect_and_plot)
        controls.addWidget(self.detect_btn, 3, 0, 1, 2)

        self.batch_btn = QtWidgets.QPushButton("Compare all records")
        self.batch_btn.clicked.connect(self.compare_all_records)
        controls.addWidget(self.batch_btn, 3, 2, 1, 2)

        # Metrics
        self.metrics_label = QtWidgets.QLabel("Select a record and click Detect && Plot.")
        self.metrics_label.setStyleSheet("font-weight: bold;")
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
            self.metrics_label.setText(f"{len(records)} records loaded. Pick one and Detect && Plot.")
        else:
            self.metrics_label.setText(f"No *{ECG_SUFFIX} files in {self.dataset_dir}")
        self._update_nav_buttons()

    def step_record(self, delta):
        """Move the record selection by +/-1 (clamped) and re-run detection."""
        n = self.record_combo.count()
        if n == 0:
            return
        i = min(n - 1, max(0, self.record_combo.currentIndex() + delta))
        if i != self.record_combo.currentIndex():
            self.record_combo.setCurrentIndex(i)
            self.detect_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        """Disable Prev/Next at the ends of the record list."""
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
        peaks = detect_r_peaks(ecg, fs)
        ref_hr = load_reference_hr(self.dataset_dir, record)

        # Overall HR from peak count over the record duration.
        duration = time[-1] - time[0]
        ecg_hr = len(peaks) / duration * 60 if duration > 0 else float("nan")
        err = abs(ecg_hr - ref_hr) if np.isfinite(ref_hr) else float("nan")

        self.metrics_label.setText(
            f"{record} | {source} | fs {fs:.0f} Hz |  R-peaks: {len(peaks)}  |  "
            f"ECG HR: {ecg_hr:.2f} bpm   Ground-truth HR: {ref_hr:.2f} bpm   |Err|: {err:.2f} bpm"
        )

        # Beat-to-beat (instantaneous) HR from RR intervals.
        peak_times = time[peaks]
        if len(peak_times) >= 2:
            rr = np.diff(peak_times)
            inst_hr = 60.0 / rr
            hr_times = (peak_times[:-1] + peak_times[1:]) / 2
        else:
            inst_hr, hr_times = np.array([]), np.array([])

        # ---- Plot ----
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        ax1.plot(time, ecg, lw=0.7, color="tab:blue", label=f"ECG ({source})")
        if len(peaks) > 0:
            ax1.scatter(time[peaks], ecg[peaks], s=18, c="red",
                        marker="x", zorder=3, label="R-peaks")
        ax1.set_title(f"{record} - ECG with detected R-peaks")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper right")

        if len(hr_times) > 0:
            ax2.plot(hr_times, inst_hr, "o-", color="tab:red", label="ECG HR (beat-to-beat)")
        ax2.axhline(ecg_hr, ls=":", color="tab:red", alpha=0.7, label=f"ECG mean HR = {ecg_hr:.1f}")
        if np.isfinite(ref_hr):
            ax2.axhline(ref_hr, ls="--", color="tab:blue", label=f"Ground-truth HR = {ref_hr:.0f}")
        ax2.set_title("Heart rate: ECG vs ground truth")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("HR (bpm)")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best")

        # Remember context so the view-window spinboxes can crop without re-detecting.
        self._ecg_ax = ax1
        self._ecg_tmax = float(time[-1])
        self._apply_view_window()

        self.canvas.draw_idle()

    def _apply_view_window(self):
        """Set the ECG subplot x-limits from the view-window spinboxes."""
        if self._ecg_ax is None:
            return
        start = max(0.0, self.view_start_spin.value())
        width = self.view_width_spin.value()
        end = min(start + width, self._ecg_tmax)
        if start >= end:  # out of range -> show the whole record
            start, end = 0.0, self._ecg_tmax
        self._ecg_ax.set_xlim(start, end)

    def update_view(self):
        """Live-update only the ECG view window (no re-detection)."""
        if self._ecg_ax is None:
            return
        self._apply_view_window()
        self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def compare_all_records(self):
        records = list_records(self.dataset_dir)
        if not records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return

        source = self.source_combo.currentText()
        self._ecg_ax = None  # batch plot replaces the axes; disable view-window edits

        progress = QtWidgets.QProgressDialog(
            "Running detection on all records...", "Cancel", 0, len(records), self)
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
            peaks = detect_r_peaks(ecg, fs)
            duration = time[-1] - time[0]
            ecg_hr = len(peaks) / duration * 60 if duration > 0 else np.nan
            rows.append({"record": record, "ecg_hr": ecg_hr, "ref_hr": ref_hr,
                         "abs_err": abs(ecg_hr - ref_hr) if np.isfinite(ref_hr) else np.nan})
            progress.setValue(i + 1)
        progress.close()

        res = pd.DataFrame(rows)
        valid = res.dropna(subset=["ref_hr", "abs_err"])
        if valid.empty:
            QtWidgets.QMessageBox.warning(self, "No results", "No valid comparisons produced.")
            return

        mae = valid["abs_err"].mean()
        rmse = np.sqrt((valid["abs_err"] ** 2).mean())
        bias = (valid["ecg_hr"] - valid["ref_hr"]).mean()
        corr = np.corrcoef(valid["ecg_hr"], valid["ref_hr"])[0, 1] if len(valid) > 1 else np.nan

        out = self.dataset_dir.parent / f"hr_batch_{source.lower()}.csv"
        res.to_csv(out, index=False)

        self.metrics_label.setText(
            f"Batch [{source}]  records: {len(valid)}  "
            f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  bias: {bias:+.2f}  r: {corr:.3f}  "
            f"(saved {out.name})"
        )

        # ---- Plot ----
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        x = valid["ref_hr"].values
        y = valid["ecg_hr"].values
        ax1.scatter(x, y, s=22, alpha=0.75, color="tab:blue")
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax1.plot([lo, hi], [lo, hi], "--", color="tab:red", lw=1, label="y = x")
        ax1.set_xlabel("Ground-truth HR (bpm)")
        ax1.set_ylabel("ECG HR (bpm)")
        ax1.set_title(f"ECG vs ground-truth HR - all records ({source})")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)

        idx = np.arange(len(res))
        ax2.bar(idx, res["abs_err"].values, color="tab:orange", alpha=0.85)
        ax2.axhline(mae, ls="--", color="tab:red", label=f"MAE = {mae:.2f} bpm")
        ax2.set_xticks(idx)
        ax2.set_xticklabels([short_label(r) for r in res["record"]],
                            rotation=90, fontsize=7)
        ax2.set_xlabel("Record")
        ax2.set_ylabel("Abs error (bpm)")
        ax2.set_title("Per-record absolute HR error")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3, axis="y")

        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PeakGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
