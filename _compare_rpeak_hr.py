"""Compare R-peak detection and HR between pan_tompkins.py and 1.py on the same records.

Both detectors are fed the *same* input column so the difference is purely algorithmic.
- PT  : pan_tompkins.detect_r_peaks  (Pan-Tompkins, adaptive threshold)
- WT  : faithful copy of 1.py calculate_heart_rate detection (wavelet db4 + find_peaks)

HR conventions reported:
- HR_count = len(peaks)/duration*60     (how ecg_peak_gui.py reports PT HR)
- HR_rr    = 60 / mean(valid RR)         (how 1.py reports HR; RR kept in 0.3-2.0 s)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks

from ECG.pan_tompkins import detect_r_peaks

DATASET = Path("dataset_primer_1")
ECG_SUFFIX = "_ecg.csv"
SOURCE_COL = "ECG_Filtered (V)"   # both algorithms run on filtered data in practice


def wt_detect(ecg, fs):
    """R-peak indices using 1.py's wavelet + find_peaks approach."""
    wavelet, level = "db4", 7
    coeffs = pywt.wavedec(ecg, wavelet, level=level)
    coeffs_denoised = [np.zeros_like(c) if i < 3 or i > 5 else c
                       for i, c in enumerate(coeffs)]
    qrs = pywt.waverec(coeffs_denoised, wavelet)[:len(ecg)]
    min_distance = int(0.25 * fs)
    mx = np.max(qrs)
    peaks, _ = find_peaks(qrs, distance=min_distance,
                          height=0.3 * mx, prominence=0.5 * mx)
    return np.asarray(peaks, dtype=int)


def hr_count(peaks, duration):
    return len(peaks) / duration * 60 if duration > 0 else np.nan


def hr_rr(peaks, fs):
    """1.py HR: mean of RR intervals filtered to 0.3-2.0 s."""
    if len(peaks) < 2:
        return np.nan
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if rr.size == 0:
        return np.nan
    return 60.0 / np.mean(rr)


def ref_hr(record):
    p = DATASET / f"{record}_metadata.json"
    if not p.exists():
        return np.nan
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return float(meta["ground_truth"]["heart_rate_bpm"])
    except Exception:
        return np.nan


def short(record):
    s = re.sub(r"_\d{8}_\d{6}$", "", record)
    return s[len("data_"):] if s.startswith("data_") else s


def main():
    records = [f.name[:-len(ECG_SUFFIX)] for f in sorted(DATASET.glob(f"*{ECG_SUFFIX}"))]
    rows = []
    for rec in records:
        df = pd.read_csv(DATASET / f"{rec}{ECG_SUFFIX}")
        df.columns = df.columns.str.strip()
        t = df["Time (s)"].values
        ecg = np.array(df[SOURCE_COL.strip()].values, dtype=float)
        dt = np.median(np.diff(t))
        fs = 1.0 / dt
        dur = t[-1] - t[0]

        pt = detect_r_peaks(ecg, fs)
        wt = wt_detect(ecg, fs)
        g = ref_hr(rec)

        rows.append({
            "record": short(rec),
            "fs": round(fs),
            "ref_hr": g,
            "pt_npeaks": len(pt),
            "wt_npeaks": len(wt),
            "pt_hr_count": hr_count(pt, dur),
            "pt_hr_rr": hr_rr(pt, fs),
            "wt_hr_count": hr_count(wt, dur),
            "wt_hr_rr": hr_rr(wt, fs),
        })

    res = pd.DataFrame(rows)
    res.to_csv("_compare_rpeak_hr_results.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)

    # -- first 15 records, side by side (HR_rr is the 1.py convention) --
    show = res.head(15).copy()
    print("\n=== First 15 records (input = %s) ===" % SOURCE_COL)
    print(show[["record", "ref_hr", "pt_npeaks", "wt_npeaks",
                "pt_hr_rr", "wt_hr_rr", "pt_hr_count"]]
          .to_string(index=False,
                     formatters={"pt_hr_rr": "{:.1f}".format,
                                 "wt_hr_rr": "{:.1f}".format,
                                 "pt_hr_count": "{:.1f}".format,
                                 "ref_hr": "{:.0f}".format}))

    # -- aggregate accuracy vs ground truth over all valid records --
    v = res.dropna(subset=["ref_hr"]).copy()
    print("\n=== Aggregate over %d records with ground truth ===" % len(v))
    for name, col in [("PT  HR=60/meanRR ", "pt_hr_rr"),
                      ("PT  HR=count/dur ", "pt_hr_count"),
                      ("WT  HR=60/meanRR ", "wt_hr_rr"),
                      ("WT  HR=count/dur ", "wt_hr_count")]:
        err = (v[col] - v["ref_hr"]).dropna()
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        bias = err.mean()
        n_ok = len(err)
        corr = np.corrcoef(v.loc[err.index, col], v.loc[err.index, "ref_hr"])[0, 1]
        print(f"{name}: MAE {mae:5.2f}  RMSE {rmse:5.2f}  bias {bias:+6.2f}  "
              f"r {corr:.3f}  (n={n_ok})")

    # -- agreement between the two detectors --
    dpk = (res["pt_npeaks"] - res["wt_npeaks"])
    dhr = (res["pt_hr_rr"] - res["wt_hr_rr"]).dropna()
    print("\n=== PT vs WT agreement (same input) ===")
    print(f"peak-count diff (PT - WT): mean {dpk.mean():+.1f}  "
          f"median {dpk.median():+.0f}  max|.| {dpk.abs().max()}")
    print(f"HR_rr diff (PT - WT)     : mean {dhr.mean():+.2f} bpm  "
          f"median {dhr.median():+.2f}  max|.| {dhr.abs().max():.1f}")
    print("\nSaved full table -> _compare_rpeak_hr_results.csv")


if __name__ == "__main__":
    main()
