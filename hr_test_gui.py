"""
PyQt6 GUI to test Pan-Tompkins heart-rate calculation on the two ECG datasets.

For a selected record it runs the detector in :mod:`ecg_processor` and shows:
  1. The ECG trace with the detected R-peaks (cropped to a scrollable window).
  2. Beat-to-beat (instantaneous) heart rate over time, with the computed mean
     and the ground-truth reference overlaid for comparison.

It works on both layouts via :mod:`ecg_datasets`:
  * dataset_primer_1  -> ``*_ecg.csv``  (Raw / Filtered, 256 Hz); scalar
                         ground-truth HR from the ``*_metadata.json`` (note: the
                         primer README warns this label is "not accurate").
  * BIDMC             -> ``bidmc_csv/*_Signals.csv`` (lead II/V/AVR, 125 Hz);
                         monitor HR series from the matching ``*_Numerics.csv``.

"Evaluate all" runs the detector over the whole folder and plots computed vs.
reference HR (agreement scatter + signed-error histogram) with MAE/RMSE/bias.

Run:
    python hr_test_gui.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "PyQt6")  # pin matplotlib's Qt backend to PyQt6

import numpy as np
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import ecg_datasets as ds
import ecg_processor as ep

DEFAULT_DATASET = ds.PRIMER_DIR if Path(ds.PRIMER_DIR).exists() else "."
PRIMER_COLUMNS = {"Raw": "ECG_Raw (V)", "Filtered": "ECG_Filtered (V)"}
BIDMC_LEADS = ["II", "V", "AVR"]

SRC_COLOR = {"primer": "#1f77b4", "bidmc": "#d62728"}
TOL_BPM = 5.0  # "within tolerance" band for the agreement summary


def _load_sqa_bad(folder: Path) -> set:
    """Names the primer SQA pass flagged as low quality (``.rr_excluded.json``)."""
    f = folder / ".rr_excluded.json"
    if not f.exists():
        return set()
    try:
        with open(f, encoding="utf-8") as fh:
            return set(json.load(fh).get("sqa_bad", []))
    except (json.JSONDecodeError, OSError):
        return set()


# --------------------------------------------------------------------------- #
class HRTestGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pan-Tompkins Heart-Rate Test")
        self.resize(1320, 920)
        self.dataset_dir = Path(DEFAULT_DATASET)
        self.records = []          # [(kind, base_path, name), ...]
        self.sqa_bad = set()
        self._cur = None           # (rec, res, kind, base)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        # Dataset folder
        controls.addWidget(QtWidgets.QLabel("Dataset folder"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.dataset_dir))
        controls.addWidget(self.dataset_edit, 0, 1, 1, 6)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_dataset)
        controls.addWidget(browse, 0, 7)

        # Record + navigation
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
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_records)
        controls.addWidget(refresh, 1, 4)
        self.analyze_btn = QtWidgets.QPushButton("Analyze && Plot")
        self.analyze_btn.clicked.connect(self.analyze_and_plot)
        controls.addWidget(self.analyze_btn, 1, 5)
        self.batch_btn = QtWidgets.QPushButton("Evaluate all")
        self.batch_btn.clicked.connect(self.evaluate_all)
        controls.addWidget(self.batch_btn, 1, 6)

        # Detector + per-source signal selection
        controls.addWidget(QtWidgets.QLabel("Detector"), 2, 0)
        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(list(ep.DETECTORS))  # pan_tompkins first
        controls.addWidget(self.detector_combo, 2, 1)
        controls.addWidget(QtWidgets.QLabel("Primer source"), 2, 2)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(PRIMER_COLUMNS.keys())
        controls.addWidget(self.source_combo, 2, 3)
        controls.addWidget(QtWidgets.QLabel("BIDMC lead"), 2, 4)
        self.lead_combo = QtWidgets.QComboBox()
        self.lead_combo.addItems(BIDMC_LEADS)
        controls.addWidget(self.lead_combo, 2, 5)
        self.skip_bad_chk = QtWidgets.QCheckBox("Skip SQA-flagged (batch)")
        self.skip_bad_chk.setChecked(True)
        controls.addWidget(self.skip_bad_chk, 2, 6)

        # View window (crops the ECG panel only)
        controls.addWidget(QtWidgets.QLabel("View start (s)"), 3, 0)
        self.view_start = QtWidgets.QDoubleSpinBox()
        self.view_start.setRange(0.0, 1e6)
        self.view_start.setSingleStep(5.0)
        controls.addWidget(self.view_start, 3, 1)
        controls.addWidget(QtWidgets.QLabel("View width (s)"), 3, 2)
        self.view_width = QtWidgets.QDoubleSpinBox()
        self.view_width.setRange(2.0, 1e6)
        self.view_width.setSingleStep(5.0)
        self.view_width.setValue(15.0)
        controls.addWidget(self.view_width, 3, 3)
        self.view_start.valueChanged.connect(self._apply_view)
        self.view_width.valueChanged.connect(self._apply_view)

        # Metrics line
        self.metrics_label = QtWidgets.QLabel("Select a record and click Analyze && Plot.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.metrics_label)

        # Figure
        self.figure = Figure(figsize=(12, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas)

        self._ecg_ax = None
        self._ecg_tmax = 0.0
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
            self.records = []
            return
        self.records = ds.discover_records(str(self.dataset_dir))
        self.sqa_bad = _load_sqa_bad(self.dataset_dir)
        self.record_combo.addItems([ds.short_label(n) for _, _, n in self.records])
        if self.records:
            kinds = {k for k, _, _ in self.records}
            self.metrics_label.setText(
                f"{len(self.records)} records ({', '.join(sorted(kinds))}). "
                f"Pick one and Analyze && Plot, or Evaluate all.")
        else:
            self.metrics_label.setText(
                f"No '*_ecg.csv' or '*_Signals.csv' records under {self.dataset_dir}")
        self._update_nav_buttons()

    def step_record(self, delta):
        n = self.record_combo.count()
        if n == 0:
            return
        i = min(n - 1, max(0, self.record_combo.currentIndex() + delta))
        if i != self.record_combo.currentIndex():
            self.record_combo.setCurrentIndex(i)
            self.analyze_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n, i = self.record_combo.count(), self.record_combo.currentIndex()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)

    # ----------------------------------------------------------------- #
    def _load(self, kind, base):
        if kind == "primer":
            col = PRIMER_COLUMNS[self.source_combo.currentText()]
            return ds.load_primer_record(base, column=col)
        return ds.load_bidmc_record(base, lead=self.lead_combo.currentText())

    def analyze_and_plot(self):
        i = self.record_combo.currentIndex()
        if i < 0 or i >= len(self.records):
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return
        kind, base, _ = self.records[i]
        try:
            rec = self._load(kind, base)
            res = ep.analyze(rec.ecg, rec.fs,
                             method=self.detector_combo.currentText(), time=rec.time)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Analyze failed", str(exc))
            return
        self._cur = (rec, res, kind, base)
        self.view_start.setMaximum(max(0.0, rec.duration - self.view_width.value()))
        self._draw()

    # ----------------------------------------------------------------- #
    def _draw(self):
        rec, res, kind, base = self._cur
        name = ds.short_label(rec.name)
        t, ecg = rec.time, rec.ecg
        peaks = res.r_peaks
        hr, ref = res.heart_rate, rec.ref_hr

        # Metrics line
        flagged = " ⚠SQA-flagged" if rec.name in self.sqa_bad else ""
        hr_str = f"{hr:.1f}" if np.isfinite(hr) else "n/a"
        ref_str = f"{ref:.1f}" if ref is not None and np.isfinite(ref) else "n/a"
        if ref is not None and np.isfinite(ref) and np.isfinite(hr):
            err = hr - ref
            err_str = f"  |  err {err:+.1f} bpm ({abs(err) / ref * 100:.1f}%)"
        else:
            err_str = ""
        ref_lbl = "metadata" if kind == "primer" else "monitor"
        self.metrics_label.setStyleSheet("font-weight: bold;")
        self.metrics_label.setText(
            f"{name}  [{rec.source}/{rec.lead}]  fs {rec.fs:.0f} Hz  "
            f"{rec.duration:.0f} s  {res.num_beats} beats{flagged}\n"
            f"Pan-Tompkins HR {hr_str} bpm  |  ref ({ref_lbl}) {ref_str} bpm{err_str}")

        self.figure.clear()
        ax_ecg, ax_hr = self.figure.subplots(2, 1)

        # 1) ECG with detected R-peaks
        ax_ecg.plot(t[:ecg.size], ecg, lw=0.7, color="tab:blue", label="ECG")
        if peaks.size:
            ax_ecg.scatter(t[peaks], ecg[peaks], s=32, c="tab:red", marker="x",
                           zorder=3, label=f"R-peaks ({peaks.size})")
        ax_ecg.set_title(f"{name} — {rec.source} lead {rec.lead}")
        ax_ecg.set_xlabel("Time (s)")
        ax_ecg.set_ylabel(rec.lead or "ECG")
        ax_ecg.grid(alpha=0.3)
        ax_ecg.legend(loc="upper right", fontsize=8)
        self._ecg_ax = ax_ecg
        self._ecg_tmax = float(t[min(ecg.size, t.size) - 1]) if t.size else 0.0
        self._apply_view(redraw=False)

        # 2) beat-to-beat HR vs computed mean and reference
        if res.instantaneous_hr.size:
            ax_hr.plot(res.hr_times, res.instantaneous_hr, "-o", ms=3, lw=0.8,
                       color="tab:green", label="beat-to-beat HR")
        if np.isfinite(hr):
            ax_hr.axhline(hr, color="tab:purple", lw=1.6,
                          label=f"mean Pan-Tompkins ({hr:.1f})")
        if kind == "bidmc":
            tn, hrn = ds.bidmc_numerics_hr(base)
            if tn is not None:
                ax_hr.plot(tn, hrn, lw=1.4, color="tab:red", alpha=0.8,
                           label=f"monitor HR ({np.mean(hrn):.1f})")
        elif ref is not None and np.isfinite(ref):
            ax_hr.axhline(ref, color="tab:red", ls="--", lw=1.4,
                          label=f"ground truth ({ref:.1f})")
        ax_hr.set_title("Heart rate")
        ax_hr.set_xlabel("Time (s)")
        ax_hr.set_ylabel("HR (bpm)")
        ax_hr.grid(alpha=0.3)
        ax_hr.legend(loc="upper right", fontsize=8)
        if t.size:
            ax_hr.set_xlim(0, self._ecg_tmax)

        self.canvas.draw_idle()

    def _apply_view(self, redraw=True):
        if self._ecg_ax is None:
            return
        start = max(0.0, self.view_start.value())
        end = min(start + self.view_width.value(), self._ecg_tmax)
        if start >= end:
            start, end = 0.0, self._ecg_tmax
        self._ecg_ax.set_xlim(start, end)
        if redraw:
            self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def evaluate_all(self):
        if not self.records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return
        method = self.detector_combo.currentText()
        skip_bad = self.skip_bad_chk.isChecked()

        progress = QtWidgets.QProgressDialog(
            "Analyzing all records...", "Cancel", 0, len(self.records), self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Evaluate all")
        progress.show()

        rows = []          # (source, ref_hr, computed_hr)
        n_no_ref = n_skipped = n_failed = 0
        for k, (kind, base, name) in enumerate(self.records):
            if progress.wasCanceled():
                break
            progress.setValue(k)
            QtWidgets.QApplication.processEvents()
            if skip_bad and name in self.sqa_bad:
                n_skipped += 1
                continue
            try:
                rec = self._load(kind, base)
                hr = ep.heart_rate(rec.ecg, rec.fs, method=method)
            except Exception:
                n_failed += 1
                continue
            if rec.ref_hr is None or not np.isfinite(rec.ref_hr) or not np.isfinite(hr):
                n_no_ref += 1
                continue
            rows.append((rec.source, rec.ref_hr, hr))
        progress.setValue(len(self.records))
        progress.close()

        if not rows:
            QtWidgets.QMessageBox.warning(
                self, "No results",
                "No records produced a computed HR with a reference to compare.")
            return

        ref = np.array([r[1] for r in rows])
        comp = np.array([r[2] for r in rows])
        srcs = [r[0] for r in rows]
        err = comp - ref
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        within = float(np.mean(np.abs(err) <= TOL_BPM) * 100)

        # Per-source MAE (primer's label is unreliable; keep them distinguishable)
        per_src = []
        for s in ("bidmc", "primer"):
            m = np.array([x == s for x in srcs])
            if m.any():
                per_src.append(f"{s} n={int(m.sum())} MAE {np.mean(np.abs(err[m])):.1f}")
        skipped_note = f", skipped {n_skipped} SQA-flagged" if n_skipped else ""
        self.metrics_label.setStyleSheet("font-weight: bold;")
        self.metrics_label.setText(
            f"[{method}] {len(rows)} records compared "
            f"(no ref {n_no_ref}, failed {n_failed}{skipped_note})  |  "
            f"MAE {mae:.1f}  RMSE {rmse:.1f}  bias {bias:+.1f} bpm  |  "
            f"within ±{TOL_BPM:.0f} bpm: {within:.0f}%   ·   " + "  ·  ".join(per_src))

        self.figure.clear()
        ax1, ax2 = self.figure.subplots(1, 2)

        # Agreement scatter: computed vs reference, with the identity line
        for s in set(srcs):
            m = np.array([x == s for x in srcs])
            ax1.scatter(ref[m], comp[m], s=30, alpha=0.75, edgecolors="0.3", lw=0.3,
                        color=SRC_COLOR.get(s, "0.5"), label=f"{s} (n={int(m.sum())})")
        lo = float(min(ref.min(), comp.min())) - 5
        hi = float(max(ref.max(), comp.max())) + 5
        ax1.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="identity")
        ax1.fill_between([lo, hi], [lo - TOL_BPM, hi - TOL_BPM],
                         [lo + TOL_BPM, hi + TOL_BPM], color="0.8", alpha=0.4,
                         label=f"±{TOL_BPM:.0f} bpm")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_xlabel("Reference HR (bpm)")
        ax1.set_ylabel("Pan-Tompkins HR (bpm)")
        ax1.set_title("Heart Rate Comparison (Hi-Me! vs Reference)")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left", fontsize=8)

        # Signed-error histogram
        ax2.hist(err, bins=max(8, int(np.sqrt(len(err)) * 2)),
                 color="tab:blue", alpha=0.8, edgecolor="0.3")
        ax2.axvline(0, color="k", lw=0.9)
        ax2.axvline(bias, color="tab:purple", lw=1.5, label=f"bias {bias:+.1f}")
        ax2.set_xlabel("Heart rate error (bpm)")
        ax2.set_ylabel("Records")
        ax2.set_title(f"Error distribution (MAE {mae:.1f}, RMSE {rmse:.1f})")
        ax2.grid(alpha=0.3, axis="y")
        ax2.legend(loc="upper right", fontsize=8)

        self._ecg_ax = None  # batch view replaces the per-record axes
        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = HRTestGui()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
