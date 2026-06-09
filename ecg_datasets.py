"""
ECG record loaders for the two datasets the SQA engine is tested on.

  * dataset_primer_1 (Hi-Me! 2.0)  -> ``*_ecg.csv``  (ECG_Raw / ECG_Filtered, 256 Hz)
                                      with a ``*_metadata.json`` ground-truth HR.
  * BIDMC PPG & Respiration         -> ``bidmc_csv/*_Signals.csv`` (lead II, 125 Hz).

Both are bridged to one :class:`ECGRecord` interface so the GUI and batch
evaluator can treat them identically.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PRIMER_DIR = os.path.join(_ROOT, "dataset_primer_1")
BIDMC_DIR = os.path.join(_ROOT, "bidmc-ppg-and-respiration-dataset")
BIDMC_CSV_DIR = os.path.join(BIDMC_DIR, "bidmc_csv")

PRIMER_SUFFIX = "_ecg.csv"
BIDMC_SUFFIX = "_Signals.csv"


@dataclass
class ECGRecord:
    name: str
    ecg: np.ndarray             # single-lead ECG samples
    fs: float
    time: np.ndarray            # per-sample time axis (s)
    ref_hr: Optional[float]     # ground-truth heart rate (bpm), if available
    source: str                 # "primer" | "bidmc"
    lead: str = ""              # which lead/column the ECG came from

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0

def short_label(name: str) -> str:
    """Trim ``data_`` prefix and ``_YYYYMMDD_HHMMSS`` timestamp from a name."""
    s = re.sub(r"_\d{8}_\d{6}$", "", name)
    return s[len("data_"):] if s.startswith("data_") else s

def _sampling_rate(time: np.ndarray) -> float:
    """fs from the total span, robust to per-sample timestamp rounding.

    (BIDMC's CSV rounds Time to 2 dp, so per-sample diffs alternate 0.008/0.01
    and the median is misleading; the endpoints stay accurate, so span wins.)
    """
    if time.size < 2:
        return float("nan")
    span = float(time[-1] - time[0])
    return (time.size - 1) / span if span > 0 else float("nan")

def _primer_ref_hr(meta_path: str) -> Optional[float]:
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return float(meta["ground_truth"]["heart_rate_bpm"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None

def load_primer_record(path: str, *, column: str = "ECG_Raw (V)") -> ECGRecord:
    """Load one primer ``*_ecg.csv``. ``path`` may include or omit the suffix."""
    base = path[:-len(PRIMER_SUFFIX)] if path.endswith(PRIMER_SUFFIX) else path
    csv_path = base + PRIMER_SUFFIX
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    col = column.strip()
    if col not in df.columns:                       # fall back to filtered/first
        col = "ECG_Filtered (V)" if "ECG_Filtered (V)" in df.columns else df.columns[-1]
    time = df["Time (s)"].to_numpy(dtype=float)
    ecg = df[col].to_numpy(dtype=float)
    fs = _sampling_rate(time)
    name = os.path.basename(base)
    ref_hr = _primer_ref_hr(base + "_metadata.json")
    return ECGRecord(name=name, ecg=ecg, fs=fs, time=time, ref_hr=ref_hr,
                     source="primer", lead=col)

def load_primer(folder: str = PRIMER_DIR, *, limit: Optional[int] = None,
                column: str = "ECG_Raw (V)") -> List[ECGRecord]:
    files = sorted(glob.glob(os.path.join(folder, f"*{PRIMER_SUFFIX}")))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        try:
            out.append(load_primer_record(f, column=column))
        except Exception:
            continue
    return out

 #

def bidmc_numerics_hr(base_path: str):
    """Monitor-reported HR series from a BIDMC ``*_Numerics.csv`` (1 Hz).

    ``base_path`` is the record base without the ``_Signals.csv`` suffix (as
    returned by :func:`discover_records`); the numerics file sits beside it.
    Returns ``(time_s, hr_bpm)`` with non-physiological samples dropped, or
    ``(None, None)`` if the file or HR column is missing.
    """
    num_path = base_path + "_Numerics.csv"
    if not os.path.exists(num_path):
        return None, None
    try:
        df = pd.read_csv(num_path)
        df.columns = df.columns.str.strip()          # header has leading spaces
        if "HR" not in df.columns:
            return None, None
        time_col = next((c for c in df.columns if c.lower().startswith("time")),
                        df.columns[0])
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        hr = pd.to_numeric(df["HR"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(hr) & (hr > 0)
        return (t[ok], hr[ok]) if ok.any() else (None, None)
    except Exception:
        return None, None

def bidmc_ref_hr(base_path: str) -> Optional[float]:
    """Mean monitor-reported HR (bpm) for a BIDMC record, or ``None``."""
    _, hr = bidmc_numerics_hr(base_path)
    return float(np.mean(hr)) if hr is not None and hr.size else None

def load_bidmc_record(path: str, *, lead: str = "II") -> ECGRecord:
    """Load one BIDMC ``*_Signals.csv``. ``path`` may include or omit the suffix."""
    base = path[:-len(BIDMC_SUFFIX)] if path.endswith(BIDMC_SUFFIX) else path
    csv_path = base + BIDMC_SUFFIX
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()             # header has leading spaces
    time_col = next((c for c in df.columns if c.lower().startswith("time")), df.columns[0])
    if lead not in df.columns:                       # prefer II, else V/AVR
        lead = next((c for c in ("II", "V", "AVR") if c in df.columns), df.columns[-1])
    time = df[time_col].to_numpy(dtype=float)
    ecg = df[lead].to_numpy(dtype=float)
    fs = _sampling_rate(time)
    name = os.path.basename(base)
    return ECGRecord(name=name, ecg=ecg, fs=fs, time=time, ref_hr=bidmc_ref_hr(base),
                     source="bidmc", lead=lead)

def load_bidmc(folder: str = BIDMC_CSV_DIR, *, limit: Optional[int] = None,
               lead: str = "II") -> List[ECGRecord]:
    files = sorted(glob.glob(os.path.join(folder, f"*{BIDMC_SUFFIX}")))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        try:
            out.append(load_bidmc_record(f, lead=lead))
        except Exception:
            continue
    return out

def discover_records(folder: str):
    """Return ``[(kind, base_path, name), ...]`` for whichever layout is present.

    Looks for primer ``*_ecg.csv`` first, then BIDMC ``*_Signals.csv`` (also
    descending into a ``bidmc_csv`` subfolder if the parent was selected).
    """
    recs = []
    for f in sorted(glob.glob(os.path.join(folder, f"*{PRIMER_SUFFIX}"))):
        base = f[:-len(PRIMER_SUFFIX)]
        recs.append(("primer", base, os.path.basename(base)))
    search_dirs = [folder]
    sub = os.path.join(folder, "bidmc_csv")
    if os.path.isdir(sub):
        search_dirs.append(sub)
    for d in search_dirs:
        for f in sorted(glob.glob(os.path.join(d, f"*{BIDMC_SUFFIX}"))):
            base = f[:-len(BIDMC_SUFFIX)]
            recs.append(("bidmc", base, os.path.basename(base)))
    return recs

def load_record(kind: str, base_path: str) -> ECGRecord:
    if kind == "primer":
        return load_primer_record(base_path)
    return load_bidmc_record(base_path)
