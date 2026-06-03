"""ecg_analysis - reusable ECG R-peak detection and heart-rate library.

Single home for the two interchangeable R-peak detectors and the shared
heart-rate helpers, extracted from ``pan_tompkins.py`` and ``1.py`` so the GUIs
and analysis scripts all share one implementation.

Detectors
---------
``"pan_tompkins"`` : adaptive double-threshold Pan-Tompkins
    (delegates to :func:`pan_tompkins.detect_r_peaks`).
``"wavelet"``      : db4 wavelet QRS-band reconstruction + ``find_peaks``
    (the method used by ``1.py``'s ``calculate_heart_rate``).

Quick start
-----------
>>> import ecg_analysis as eca
>>> res = eca.analyze(ecg, fs, method="pan_tompkins")
>>> res.heart_rate          # mean HR in bpm
>>> res.r_peaks             # R-peak sample indices

Heart rate is computed as ``60 / mean(valid RR)`` with RR intervals filtered to
0.3-2.0 s; this is more robust than peak-count / duration.

Note: respiratory rate (1.py's PSD/MPF step) is intentionally left out -- it
depends on the project's ``utils`` module and is a separate concern from R-peak
and HR estimation.
"""
from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import find_peaks

from ECG.pan_tompkins import detect_r_peaks as _detect_pan_tompkins
from ECG.pan_tompkins import heart_rate_from_peaks

__all__ = [
    "detect_r_peaks",
    "detect_r_peaks_pan_tompkins",
    "detect_r_peaks_wavelet",
    "heart_rate_from_peaks",
    "instantaneous_heart_rate",
    "analyze",
    "ECGAnalysis",
    "DETECTORS",
]

# Re-export under an explicit name so callers don't import pan_tompkins directly.
detect_r_peaks_pan_tompkins = _detect_pan_tompkins


def detect_r_peaks_wavelet(ecg, fs, wavelet="db4", level=7,
                           height_frac=0.3, prominence_frac=0.5):
    """Detect R-peak indices via discrete wavelet reconstruction.

    Decomposes the signal with ``wavelet`` to ``level`` levels, keeps only the
    QRS-band detail coefficients (indices 3-5), reconstructs, then selects peaks
    at least 0.25 s apart whose height and prominence exceed ``height_frac`` /
    ``prominence_frac`` of the reconstruction's maximum. Mirrors the detector in
    ``1.py``'s ``calculate_heart_rate``.
    """
    ecg = np.array(ecg, dtype=float)  # writable copy: pywt rejects read-only input
    coeffs = pywt.wavedec(ecg, wavelet, level=level)
    kept = [c if 3 <= i <= 5 else np.zeros_like(c) for i, c in enumerate(coeffs)]
    qrs = pywt.waverec(kept, wavelet)[:len(ecg)]
    mx = float(np.max(qrs))
    peaks, _ = find_peaks(qrs, distance=int(0.25 * fs),
                          height=height_frac * mx,
                          prominence=prominence_frac * mx)
    return np.asarray(peaks, dtype=int)


# Registry so callers can pick a detector by name.
DETECTORS = {
    "pan_tompkins": detect_r_peaks_pan_tompkins,
    "wavelet": detect_r_peaks_wavelet,
}


def detect_r_peaks(ecg, fs, method="pan_tompkins"):
    """Detect R-peak indices using ``method`` (see :data:`DETECTORS`)."""
    try:
        detector = DETECTORS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(DETECTORS)}") from None
    return detector(ecg, fs)


def instantaneous_heart_rate(peaks, fs, time=None, rr_min=0.3, rr_max=2.0):
    """Beat-to-beat heart rate.

    Returns ``(times_s, hr_bpm)`` evaluated at the midpoint of each RR interval.
    If ``time`` (the per-sample time axis) is given, peak times are taken from it
    so the result lines up with a plotted signal; otherwise they are ``peaks/fs``.
    Intervals outside ``[rr_min, rr_max]`` seconds are dropped.
    """
    peaks = np.asarray(peaks)
    if peaks.size < 2:
        return np.array([]), np.array([])
    t = peaks / fs if time is None else np.asarray(time)[peaks]
    rr = np.diff(t)
    mid = (t[:-1] + t[1:]) / 2
    ok = (rr > rr_min) & (rr < rr_max)
    return mid[ok], 60.0 / rr[ok]


@dataclass
class ECGAnalysis:
    """Outcome of :func:`analyze` for one detector on one signal."""
    method: str
    fs: float
    r_peaks: np.ndarray            # R-peak sample indices
    heart_rate: float             # bpm, mean of valid RR intervals
    hr_times: np.ndarray          # s, midpoints for the beat-to-beat series
    instantaneous_hr: np.ndarray  # bpm, one value per beat

    @property
    def num_beats(self):
        return int(len(self.r_peaks))


def analyze(ecg, fs, method="pan_tompkins", time=None):
    """Detect R peaks and compute heart rate, returning an :class:`ECGAnalysis`."""
    peaks = detect_r_peaks(ecg, fs, method)
    hr = heart_rate_from_peaks(peaks, fs)
    hr_t, inst = instantaneous_heart_rate(peaks, fs, time=time)
    return ECGAnalysis(method=method, fs=float(fs), r_peaks=peaks,
                       heart_rate=hr, hr_times=hr_t, instantaneous_hr=inst)
