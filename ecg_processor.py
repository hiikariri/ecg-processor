"""
ECG R-peak detection and heart-rate library (Pan-Tompkins and wavelet).

Usage:
    res = analyze(ecg, fs)   # method="pan_tompkins" (default) or "wavelet"
    res.heart_rate           # mean HR (bpm)
    res.r_peaks              # R-peak sample indices
"""
from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks

__all__ = [
    "bandpass_filter",
    "detect_r_peaks",
    "detect_r_peaks_pan_tompkins",
    "detect_r_peaks_wavelet",
    "heart_rate_from_peaks",
    "instantaneous_heart_rate",
    "heart_rate",
    "analyze",
    "ECGAnalysis",
    "DETECTORS",
]

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply a zero-phase Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)

def _integrate(ecg, fs):
    """Pan-Tompkins preprocessing: BPF (5-15 Hz) -> derivative -> square ->
    150 ms moving-window integration. Returns the integrated signal."""
    filtered = bandpass_filter(ecg, lowcut=5, highcut=15, fs=fs)

    derivative = np.zeros_like(filtered)
    derivative[1:] = np.diff(filtered)

    squared = derivative ** 2

    window = max(1, int(0.150 * fs))
    return np.convolve(squared, np.ones(window) / window, mode="same")

def _detect_peaks(mwa, fs):
    """Adaptive double-threshold detector on the integrated signal.

    Maintains running signal (SPKI) and noise (NPKI) level estimates with the
    1/8 : 7/8 update rule, where THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI) and
    THRESHOLD_I2 = 0.5 * THRESHOLD_I1 is the search-back threshold. If no QRS is
    found within 1.66x the recent average RR, the interval is re-scanned
    against I2. Returns indices into ``mwa``.
    """
    n = len(mwa)
    if n < 3:
        return np.array([], dtype=int)

    min_distance = int(0.25 * fs)

    # Learning phase: seed signal/noise levels from the first 2 s. Seed the
    # signal level (SPKI) from the average of the local maxima in that window
    # rather than its single largest sample, so a filter start-up transient
    # can't push the threshold above every real QRS and stall detection.
    learn = mwa[: int(2 * fs)] if n > int(2 * fs) else mwa
    interior = learn[1:-1]
    learn_peaks = interior[(interior > learn[:-2]) & (interior > learn[2:])]
    SPKI = float(np.mean(learn_peaks)) if learn_peaks.size else float(np.max(learn))
    NPKI = float(np.mean(learn))
    threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
    threshold_I2 = 0.5 * threshold_I1

    signal_peaks = []
    peaks = []        # every local maximum, in order
    indexes = []      # position within `peaks` of each accepted signal peak
    RR_missed = 0
    index = 0

    for i in range(1, n - 1):
        if mwa[i - 1] < mwa[i] and mwa[i + 1] < mwa[i]:
            peaks.append(i)

            far_enough = (not signal_peaks) or (i - signal_peaks[-1]) > 0.3 * fs
            if mwa[i] > threshold_I1 and far_enough:
                signal_peaks.append(i)
                indexes.append(index)
                SPKI = 0.125 * mwa[i] + 0.875 * SPKI

                # Search-back: recover a beat missed between the last two peaks.
                if RR_missed != 0 and len(signal_peaks) >= 2:
                    if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                        section = peaks[indexes[-2] + 1: indexes[-1]]
                        cand = [p for p in section
                                if p - signal_peaks[-2] > min_distance
                                and signal_peaks[-1] - p > min_distance
                                and mwa[p] > threshold_I2]
                        if cand:
                            best = cand[int(np.argmax(mwa[cand]))]
                            signal_peaks.append(signal_peaks[-1])
                            signal_peaks[-2] = best
            else:
                NPKI = 0.125 * mwa[i] + 0.875 * NPKI

            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_I2 = 0.5 * threshold_I1

            if len(signal_peaks) > 8:
                RR_ave = int(np.mean(np.diff(signal_peaks[-9:])))
                RR_missed = int(1.66 * RR_ave)

            index += 1

    return np.array(sorted(set(signal_peaks)), dtype=int)

def detect_r_peaks_pan_tompkins(ecg, fs):
    """Detect R-peak indices in an ECG signal using the Pan-Tompkins algorithm.

    Parameters
    ----------
    ecg : np.ndarray
        ECG samples.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Indices of the detected R peaks within ``ecg``.
    """
    ecg = np.asarray(ecg, dtype=float)
    mwa = _integrate(ecg, fs)
    approx_peaks = _detect_peaks(mwa, fs)

    # Refine each detection to the local maximum (the actual R peak) nearby.
    search = int(0.1 * fs)  # 100 ms search window.
    refined = [
        max(0, p - search) + int(np.argmax(ecg[max(0, p - search): p + search]))
        for p in approx_peaks
    ]
    return np.array(sorted(set(refined)), dtype=int)

def detect_r_peaks_wavelet(ecg, fs, wavelet="db4", level=7,
                           height_frac=0.3, prominence_frac=0.5):
    """Detect R-peak indices via discrete wavelet reconstruction.

    Decomposes the signal with ``wavelet`` to ``level`` levels, keeps only the
    QRS-band detail coefficients (indices 3-5), reconstructs, then selects peaks
    at least 0.25 s apart whose height and prominence exceed ``height_frac`` /
    ``prominence_frac`` of the reconstruction's maximum. Mirrors the detector in
    the original ``1.py``'s ``calculate_heart_rate``.
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

def heart_rate_from_peaks(peaks, fs, rr_min=0.3, rr_max=2.0):
    """Mean heart rate (bpm) from R-peak sample indices.

    RR intervals outside ``[rr_min, rr_max]`` seconds (i.e. below ~30 or above
    200 bpm) are dropped before averaging, so a single missed or doubled beat
    does not skew the estimate. Returns ``nan`` when fewer than two peaks -- or
    no in-range RR interval -- are available.
    """
    peaks = np.asarray(peaks)
    if peaks.size < 2:
        return float("nan")
    rr = np.diff(peaks) / fs
    rr = rr[(rr > rr_min) & (rr < rr_max)]
    if rr.size == 0:
        return float("nan")
    return 60.0 / float(np.mean(rr))

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

def heart_rate(ecg, fs, method="pan_tompkins"):
    """Detect R peaks with ``method`` and return the mean heart rate (bpm)."""
    return heart_rate_from_peaks(detect_r_peaks(ecg, fs, method), fs)

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
