import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply a zero-phase Butterworth bandpass filter."""
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


def detect_r_peaks(ecg, fs):
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
    mwa = _integrate(ecg, fs)
    approx_peaks = _detect_peaks(mwa, fs)

    # Refine each detection to the local maximum (the actual R peak) nearby.
    search = int(0.1 * fs)  # 100 ms search window.
    refined = [
        max(0, p - search) + int(np.argmax(ecg[max(0, p - search): p + search]))
        for p in approx_peaks
    ]
    return np.array(sorted(set(refined)), dtype=int)


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


def heart_rate(ecg, fs):
    """Detect R peaks with Pan-Tompkins and return the mean heart rate (bpm)."""
    return heart_rate_from_peaks(detect_r_peaks(ecg, fs), fs)
