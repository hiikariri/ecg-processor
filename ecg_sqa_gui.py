"""
GUI for the ECG Signal Quality Assessment engine (:mod:`ecg_sqa`).

Load an ECG record and watch the SQA pipeline decide whether it is usable:
  1. ECG trace with the detected R-peaks, titled with the quality verdict.
  2. Overlaid beat templates (the windows the iScore correlates) + the mean beat.
  3. The graded quality index (iScore, with kSQI for context) vs its thresholds.

Supports both layouts via :mod:`ecg_datasets`:
  * dataset_primer_1  -> ``*_ecg.csv``        (Raw / Filtered, 256 Hz)
  * BIDMC             -> ``bidmc_csv/*_Signals.csv`` (lead II, 125 Hz)

"Evaluate all" runs the whole folder and plots the quality distribution.

Run:
    python ecg_sqa_gui.py
"""
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import ecg_datasets as ds
from ecg_sqa import ECGSQAEngine

DEFAULT_DATASET = ds.PRIMER_DIR if Path(ds.PRIMER_DIR).exists() else "."
PRIMER_COLUMNS = {"Raw": "ECG_Raw (V)", "Filtered": "ECG_Filtered (V)"}

# Colours per verdict tier
TIER_COLOR = {"diag": "#1b7837", "hr": "#7fbf7b", "bad": "#d6604d", None: "#b2182b"}


def _verdict_color(res):
    if res["quality"] == "acceptable":
        return TIER_COLOR["diag"] if res["iscore_quality"] == "diag" else TIER_COLOR["hr"]
    return TIER_COLOR["bad"] if res["iscore_quality"] == "bad" else TIER_COLOR[None]


# --------------------------------------------------------------------------- #
class SQAGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Signal Quality Assessment")
        self.resize(1320, 940)
        self.dataset_dir = Path(DEFAULT_DATASET)
        self.records = []          # [(kind, base_path, name), ...]
        self._cur = None           # (rec, res, name)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        # Dataset folder
        controls.addWidget(QtWidgets.QLabel("Dataset folder"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.dataset_dir))
        controls.addWidget(self.dataset_edit, 0, 1, 1, 5)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_dataset)
        controls.addWidget(browse, 0, 6)

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

        # Primer ECG source + detector
        controls.addWidget(QtWidgets.QLabel("Primer source"), 2, 0)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(PRIMER_COLUMNS.keys())
        controls.addWidget(self.source_combo, 2, 1)
        controls.addWidget(QtWidgets.QLabel("Detector"), 2, 2)
        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(["pan_tompkins", "wavelet"])
        controls.addWidget(self.detector_combo, 2, 3)

        # View window (crops the ECG panel only)
        controls.addWidget(QtWidgets.QLabel("View start (s)"), 2, 4)
        self.view_start = QtWidgets.QDoubleSpinBox()
        self.view_start.setRange(0.0, 1e6); self.view_start.setSingleStep(5.0)
        controls.addWidget(self.view_start, 2, 5)
        controls.addWidget(QtWidgets.QLabel("View width (s)"), 2, 6)
        self.view_width = QtWidgets.QDoubleSpinBox()
        self.view_width.setRange(2.0, 1e6); self.view_width.setSingleStep(5.0)
        self.view_width.setValue(15.0)
        controls.addWidget(self.view_width, 2, 7)
        self.view_start.valueChanged.connect(self._apply_view)
        self.view_width.valueChanged.connect(self._apply_view)

        # Buttons
        self.assess_btn = QtWidgets.QPushButton("Assess && Plot")
        self.assess_btn.clicked.connect(self.assess_and_plot)
        controls.addWidget(self.assess_btn, 1, 5, 1, 1)
        self.batch_btn = QtWidgets.QPushButton("Evaluate all")
        self.batch_btn.clicked.connect(self.evaluate_all)
        controls.addWidget(self.batch_btn, 1, 6, 1, 1)

        # Verdict label
        self.metrics_label = QtWidgets.QLabel("Select a record and click Assess && Plot.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.metrics_label)

        # Figure
        self.figure = Figure(figsize=(12, 8.5), tight_layout=True)
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
        self.record_combo.addItems([ds.short_label(n) for _, _, n in self.records])
        if self.records:
            kinds = {k for k, _, _ in self.records}
            self.metrics_label.setText(
                f"{len(self.records)} records ({', '.join(sorted(kinds))}). "
                f"Pick one and Assess && Plot, or Evaluate all.")
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
            self.assess_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n, i = self.record_combo.count(), self.record_combo.currentIndex()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)

    # ----------------------------------------------------------------- #
    def _load_current(self):
        i = self.record_combo.currentIndex()
        if i < 0 or i >= len(self.records):
            return None
        kind, base, name = self.records[i]
        if kind == "primer":
            col = PRIMER_COLUMNS[self.source_combo.currentText()]
            return ds.load_primer_record(base, column=col)
        return ds.load_bidmc_record(base)

    def assess_and_plot(self):
        rec = self._load_current()
        if rec is None:
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return
        try:
            res = ECGSQAEngine(rec.ecg, rec.fs,
                               detector=self.detector_combo.currentText()).assess()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Assess failed", str(exc))
            return
        self._cur = (rec, res, ds.short_label(rec.name))
        self.view_start.setMaximum(max(0.0, rec.duration - self.view_width.value()))
        self._draw()

    # ----------------------------------------------------------------- #
    def _draw(self):
        rec, res, name = self._cur
        fs = rec.fs
        t = rec.time
        ecg = res["filtered"]  # the exact signal assess() evaluated

        # Verdict line
        isc = res["iscore"]
        sat = res['raw_sat_fraction'] if res['raw_sat_fraction'] is not None else float('nan')
        gate = (f"flat {res['flat_proportion']:.2f} | clip {res['clip_fraction']:.2f} | "
                f"sat {sat:.0%} | HR {res['mean_hr'] or 0:.0f} | "
                f"CoV {res['amp_cov'] if res['amp_cov'] is not None else float('nan'):.2f} | "
                f"kSQI {res['ksqi']:.1f}")
        verdict = res["label"].replace("\n", "  ")
        self.metrics_label.setText(
            f"{name}  [{rec.source}/{rec.lead}]  fs {fs:.0f} Hz  —  {verdict}\n{gate}")
        self.metrics_label.setStyleSheet(
            f"font-weight: bold; color: {_verdict_color(res)};")

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
        ax_ecg = self.figure.add_subplot(gs[0, :])
        ax_beats = self.figure.add_subplot(gs[1, 0])
        ax_score = self.figure.add_subplot(gs[1, 1])

        # 1) ECG with detected R-peaks
        ax_ecg.plot(t[:ecg.size], ecg, lw=0.7, color="tab:blue", label="ECG (filtered)")
        prim = res.get("peaks")
        if prim is not None and len(prim):
            ax_ecg.scatter(t[prim], ecg[prim], s=30, c="tab:red", marker="x",
                           zorder=3, label=f"R-peaks ({len(prim)})")
        ax_ecg.set_title(f"{name} — {verdict}", color=_verdict_color(res))
        ax_ecg.set_xlabel("Time (s)"); ax_ecg.set_ylabel("ECG (filtered)")
        ax_ecg.grid(alpha=0.3); ax_ecg.legend(loc="upper right", fontsize=8)
        self._ecg_ax = ax_ecg
        self._ecg_tmax = float(t[min(ecg.size, t.size) - 1])
        self._apply_view(redraw=False)

        # 2) beat templates overlay
        beats = res.get("beats")
        if beats is not None and len(beats) >= 2:
            tb = (np.arange(beats.shape[1]) - beats.shape[1] / 2) / fs * 1000.0
            for b in beats:
                ax_beats.plot(tb, b, color="0.7", lw=0.5, alpha=0.5)
            ax_beats.plot(tb, beats.mean(axis=0), color="tab:purple", lw=2,
                          label="mean beat")
            ax_beats.set_title(f"Beat templates  (n={len(beats)}, "
                               f"iScore = {isc:.3f})")
            ax_beats.set_xlabel("Time around R (ms)"); ax_beats.set_ylabel("Amp")
            ax_beats.legend(loc="upper right", fontsize=8)
        else:
            ax_beats.text(0.5, 0.5, "No beat templates\n(rejected before Stage 3)",
                          ha="center", va="center", transform=ax_beats.transAxes,
                          color="0.4")
            ax_beats.set_xticks([]); ax_beats.set_yticks([])
        ax_beats.grid(alpha=0.3)

        # 3) graded quality index: iScore (with kSQI shown for context)
        E = ECGSQAEngine
        s = isc if isc is not None else 0.0
        ksqi = res["ksqi"] if res["ksqi"] is not None else 0.0
        names = ["iScore", "kSQI/20"]              # kSQI rescaled onto 0–1 for display
        vals = [s, min(ksqi / 20.0, 1.0)]
        s_color = (TIER_COLOR["diag"] if s >= E.ISCORE_L2 else
                   (TIER_COLOR["hr"] if s >= E.ISCORE_L1 else TIER_COLOR[None]))
        colors = [s_color, "0.6"]                  # kSQI is informational (grey)
        y = np.arange(len(names))
        ax_score.barh(y, vals, color=colors, alpha=0.85)
        ax_score.set_yticks(y); ax_score.set_yticklabels(names)
        ax_score.set_xlim(0, 1.0)
        ax_score.axvline(E.ISCORE_L1, ls="--", color="0.5", lw=0.9, label=f"L1 min ({E.ISCORE_L1})")
        ax_score.axvline(E.ISCORE_L2, ls=":", color="0.5", lw=0.9, label=f"L2 diag ({E.ISCORE_L2})")
        labels = [f"{s:.2f}", f"{ksqi:.1f}"]       # annotate true values
        for yi, v, lab in zip(y, vals, labels):
            ax_score.text(min(v + 0.02, 0.93), yi, lab, va="center", fontsize=9)
        ax_score.set_title("Quality index (iScore grades; kSQI for context)")
        ax_score.set_xlabel("score (0–1, higher = better)")
        ax_score.legend(loc="lower right", fontsize=7)

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
        detector = self.detector_combo.currentText()
        col = PRIMER_COLUMNS[self.source_combo.currentText()]

        progress = QtWidgets.QProgressDialog(
            "Assessing all records...", "Cancel", 0, len(self.records), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Evaluate all")
        progress.show()

        tiers = {"diag": 0, "hr": 0, "bad": 0, "reject": 0}
        pts = []   # (iscore, ksqi, tier) for records that reached Stage 3
        for k, (kind, base, name) in enumerate(self.records):
            if progress.wasCanceled():
                break
            progress.setValue(k); QtWidgets.QApplication.processEvents()
            try:
                rec = (ds.load_primer_record(base, column=col) if kind == "primer"
                       else ds.load_bidmc_record(base))
                res = ECGSQAEngine(rec.ecg, rec.fs, detector=detector).assess()
            except Exception:
                continue
            tq = res["iscore_quality"]
            if res["quality"] == "acceptable":
                tiers[tq] += 1
            elif tq == "bad":
                tiers["bad"] += 1
            else:
                tiers["reject"] += 1
            if res["iscore"] is not None:
                pts.append((res["iscore"], res["ksqi"] or 0.0, tq))
        progress.close()

        total = sum(tiers.values())
        if total == 0:
            QtWidgets.QMessageBox.warning(self, "No results", "No records assessed.")
            return
        acc = tiers["diag"] + tiers["hr"]
        self.metrics_label.setStyleSheet("font-weight: bold;")
        self.metrics_label.setText(
            f"[{detector}] {total} records | acceptable {acc} ({100*acc/total:.0f}%): "
            f"diagnostic {tiers['diag']}, HR-quality {tiers['hr']} | "
            f"unacceptable {tiers['bad']+tiers['reject']} "
            f"(low-morphology {tiers['bad']}, rejected earlier {tiers['reject']})")

        self.figure.clear()
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        labels = ["Diagnostic", "HR quality", "Low morph.", "Rejected\n(stage 1-3)"]
        counts = [tiers["diag"], tiers["hr"], tiers["bad"], tiers["reject"]]
        bar_colors = [TIER_COLOR["diag"], TIER_COLOR["hr"], TIER_COLOR["bad"], "#777777"]
        ax1.bar(labels, counts, color=bar_colors, alpha=0.9)
        for i, c in enumerate(counts):
            ax1.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
        ax1.set_ylabel("records"); ax1.set_title(f"Quality distribution (n={total})")
        ax1.grid(alpha=0.3, axis="y")

        if pts:
            isc = np.array([p[0] for p in pts]); ks = np.array([p[1] for p in pts])
            cols = [TIER_COLOR[p[2]] for p in pts]
            ax2.scatter(isc, ks, c=cols, s=28, alpha=0.8, edgecolors="0.3", lw=0.3)
            ax2.axvline(ECGSQAEngine.ISCORE_L1, ls="--", color="0.5", lw=0.8,
                        label=f"L1 min ({ECGSQAEngine.ISCORE_L1})")
            ax2.axvline(ECGSQAEngine.ISCORE_L2, ls=":", color="0.5", lw=0.8,
                        label=f"L2 diag ({ECGSQAEngine.ISCORE_L2})")
            ax2.legend(loc="lower right", fontsize=7)
        ax2.set_xlabel("iScore (beat morphology) — grades quality")
        ax2.set_ylabel("kSQI (kurtosis) — context")
        ax2.set_xlim(min(0, ax2.get_xlim()[0]), 1.0)
        ax2.set_title("Per-record quality space")
        ax2.grid(alpha=0.3)

        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SQAGui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
