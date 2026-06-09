"""
ECG Signal Quality Assessment (SQA).

A short-circuit, multi-stage feasibility pipeline that decides whether an ECG
segment is good enough for heart-rate or diagnostic use. Each stage can reject
the signal outright; only signals that survive every stage are scored.

    Stage 1  flat line / saturation   reject dead, rail-clipped, or railed-noise
                                       signals (judged on the raw ADC samples)
    Stage 2  QRS / HR feasibility     R-peaks detectable, HR in a plausible
                                       range, QRS amplitude not wildly variable
    Stage 3  beat morphology (iScore) average inter-beat correlation; rejects
                                       inconsistent beats and tiers the
                                       survivors into HR-quality vs diagnostic

Usage
-----
    from ecg_sqa import ECGSQAEngine
    res = ECGSQAEngine(ecg, fs).assess()
    res["quality"]   # "acceptable" | "unacceptable"
    res["label"]     # human-readable verdict
"""
from __future__ import annotations

import numpy as np

from ecg_processor import (
    bandpass_filter,
    detect_r_peaks_pan_tompkins,
    detect_r_peaks_wavelet,
)

__all__ = ["ECGSQAEngine"]


class ECGSQAEngine:
    """Multi-stage ECG signal quality assessment engine."""

    # ── Tunable thresholds (override on the instance before assess()) ──────
    BANDPASS        = (0.5, 40.0)  # Hz, SQA analysis band (clipped to Nyquist)
    FLAT_THRESH     = 0.50    # fraction of near-constant pairs -> flat line
    CLIP_THRESH     = 0.20    # fraction of samples pinned to a rail -> saturated
    RAW_SAT_THRESH  = 0.40    # fraction of RAW samples at the ADC rails -> noise/saturation
    SAT_TOL         = 0.02    # rail band width, as a fraction of the raw range
    HR_MIN          = 24      # BPM lower bound (Stage 2)
    HR_MAX          = 240     # BPM upper bound (Stage 2)
    QRS_COV_THRESH  = 0.90    # max coefficient of variation of R-peak amplitudes
    ISCORE_L1       = 0.50    # iScore below this -> Unacceptable
    ISCORE_L2       = 0.80    # iScore above this -> Diagnostic (else HR quality)

    def __init__(self, ecg, fs, *, filtered=None, detector="pan_tompkins"):
        """
        Parameters
        ----------
        ecg : array-like
            Raw ECG samples (single lead).
        fs : float
            Sampling frequency in Hz.
        filtered : array-like, optional
            Pre-filtered signal to assess. If omitted, the raw signal is
            band-pass filtered with :data:`BANDPASS`.
        detector : str
            Primary detector for Stage 2/3 ("pan_tompkins" or "wavelet").
        """
        self.fs = float(fs)
        self.raw = np.asarray(ecg, dtype=float)
        self.detector = detector
        self.filtered = (np.asarray(filtered, dtype=float)
                         if filtered is not None else self._filter(self.raw))

    # ── Preprocessing ─────────────────────────────────────────────────────
    def _filter(self, sig):
        """Band-pass filter for SQA, clipping the high cut below Nyquist."""
        lo, hi = self.BANDPASS
        nyq = 0.5 * self.fs
        hi = min(hi, 0.45 * self.fs)            # stay safely below Nyquist
        if sig.size < 30 or lo <= 0 or hi <= lo:
            return sig - np.mean(sig)
        try:
            return bandpass_filter(sig, lo, hi, self.fs)
        except Exception:
            return sig - np.mean(sig)           # filter unstable -> just de-mean

    def _primary_peaks(self):
        if self.detector == "wavelet":
            return detect_r_peaks_wavelet(self.filtered, self.fs)
        return detect_r_peaks_pan_tompkins(self.filtered, self.fs)

    # ── Stage 1: flat line / saturation ───────────────────────────────────
    def check_flat_line(self):
        """Detect a dead (flat) or rail-clipped (saturated) signal.

        Returns ``(is_bad, proportion, clip_fraction, absdiff)`` where
        ``proportion`` is the fraction of near-constant consecutive pairs and
        ``clip_fraction`` is the fraction of samples pinned at the min/max rail.
        """
        sig = self.filtered
        s_min, s_max = float(sig.min()), float(sig.max())
        if s_max == s_min:
            return True, 1.0, 1.0, np.zeros(max(sig.size - 1, 0))

        sig_norm = (sig - s_min) / (s_max - s_min) * 1000.0
        absdiff = np.abs(np.diff(sig_norm))
        proportion = float(np.mean(absdiff <= 1.0))

        # clipping: many samples sitting exactly on the extreme rails
        rail = (np.isclose(sig, s_min) | np.isclose(sig, s_max))
        clip_fraction = float(np.mean(rail))

        is_bad = proportion > self.FLAT_THRESH or clip_fraction > self.CLIP_THRESH
        return is_bad, proportion, clip_fraction, absdiff

    def check_saturation(self):
        """Detect a railed / saturated *raw* acquisition (the real 'pure noise').

        Filtering removes the DC rails, so saturation must be judged on the raw
        ADC samples: a saturated capture spends a large fraction of its samples
        pinned at the extreme rails. Empirically, full-scale noise sits well
        above :data:`RAW_SAT_THRESH` (~0.54) while even ECG with tall R-peaks
        that occasionally clip stays below ~0.23.

        Returns ``(is_saturated, rail_fraction)``.
        """
        raw = self.raw
        if raw.size == 0:
            return False, 0.0
        lo, hi = float(raw.min()), float(raw.max())
        rng = hi - lo
        if rng == 0:
            return True, 1.0
        tol = self.SAT_TOL * rng
        rail = (raw <= lo + tol) | (raw >= hi - tol)
        frac = float(np.mean(rail))
        return frac > self.RAW_SAT_THRESH, frac

    # ── Stage 2: QRS / HR feasibility ─────────────────────────────────────
    def _hr_info(self, peaks):
        """Mean HR and valid RR intervals (s) from peak indices."""
        peaks = np.asarray(peaks)
        rr_min, rr_max = 60.0 / self.HR_MAX, 60.0 / self.HR_MIN
        if peaks.size < 2:
            return {"mean_hr": 0.0, "rr_intervals": np.array([])}
        rr = np.diff(peaks) / self.fs
        rr_valid = rr[(rr >= rr_min) & (rr <= rr_max)]
        mean_hr = 60.0 / float(np.mean(rr_valid)) if rr_valid.size else 0.0
        return {"mean_hr": mean_hr, "rr_intervals": rr_valid}

    def check_qrs(self, peaks, hr_info):
        """HR-range and QRS amplitude-consistency check.

        Returns ``(passes, mean_hr, amp_cov, note)``.
        """
        mean_hr = hr_info.get("mean_hr", 0.0)
        if len(peaks) < 2:
            return False, mean_hr, None, "Too few R-peaks detected"
        if mean_hr == 0.0:
            return False, mean_hr, None, (
                f"No valid HR — RR intervals outside [{self.HR_MIN}-{self.HR_MAX}] BPM")
        if not (self.HR_MIN <= mean_hr <= self.HR_MAX):
            return False, mean_hr, None, (
                f"HR {mean_hr:.1f} BPM outside [{self.HR_MIN}-{self.HR_MAX}]")

        amps = np.abs(self.filtered[peaks])
        mu = float(np.mean(amps))
        if mu == 0:
            return False, mean_hr, None, "Zero mean QRS amplitude"
        cov = float(np.std(amps) / mu)
        if cov >= self.QRS_COV_THRESH:
            return False, mean_hr, cov, f"QRS amp CoV {cov:.2f} >= {self.QRS_COV_THRESH}"
        return True, mean_hr, cov, "OK"

    # ── kurtosis SQI (informational) ──────────────────────────────────────
    def compute_ksqi(self):
        """Kurtosis SQI: clean ECG is spiky (high kurtosis); noise is ~Gaussian."""
        x = self.filtered
        sd = np.std(x)
        if sd == 0:
            return 0.0
        z = (x - np.mean(x)) / sd
        return float(np.mean(z ** 4))   # Fisher kurtosis ~3 for noise, higher for ECG

    # ── Stage 3: beats' average correlation (iScore) ──────────────────────
    def compute_iscore(self, peaks, hr_info):
        """Average inter-beat correlation (morphology consistency).

        Windows of width ~RR centred on each R-peak are correlated pairwise;
        the mean of the inter-correlation matrix is the iScore in [-1, 1].
        Returns ``(iscore, M_x, G_x, beats)`` where ``beats`` are the aligned
        QRS windows (for plotting).
        """
        if len(peaks) < 2:
            return 0.0, None, None, None
        rr = hr_info.get("rr_intervals", np.array([]))
        if len(rr) == 0:
            return 0.0, None, None, None

        rr_samp = rr * self.fs
        beta = int(round(min(float(np.mean(rr_samp)), float(np.median(rr_samp)))))
        half_win = max(1, beta // 2)

        ecg = self.filtered
        n = ecg.size
        beats = [ecg[int(pk) - half_win:int(pk) + half_win]
                 for pk in peaks
                 if int(pk) - half_win >= 0 and int(pk) + half_win <= n]
        if len(beats) < 2:
            return 0.0, None, None, None

        win_len = min(b.size for b in beats)
        Q = np.array([b[:win_len] for b in beats])     # (n_beats, win_len)
        M_x = np.corrcoef(Q)
        G_x = np.mean(M_x, axis=1)
        iscore = float(np.mean(G_x))
        return iscore, M_x, G_x, Q

    # ── Full pipeline ─────────────────────────────────────────────────────
    def assess(self):
        """Run the staged pipeline and return a result dict.

        Keys: ``quality`` ("acceptable"/"unacceptable"), ``label``,
        ``stage_reached`` (1-3), plus per-stage diagnostics
        (``flat_proportion``, ``clip_fraction``, ``raw_sat_fraction``,
        ``mean_hr``, ``amp_cov``, ``ksqi``, ``iscore``, ``iscore_quality``) and
        arrays for plotting (``absdiff_norm``, ``peaks``, ``M_x``, ``G_x``,
        ``beats``).
        """
        r = {
            "quality": "unacceptable", "label": "", "stage_reached": 0,
            "fs": self.fs, "filtered": self.filtered,
            "flat_proportion": None, "clip_fraction": None, "is_flat": False,
            "raw_sat_fraction": None, "absdiff_norm": None,
            "peaks": None, "hr_info": None, "mean_hr": None, "amp_cov": None,
            "ksqi": None, "iscore": None, "iscore_quality": None,
            "M_x": None, "G_x": None, "beats": None,
        }
        r["ksqi"] = self.compute_ksqi()

        # Stage 1 — flat line / saturation (raw rails) / clipping
        r["stage_reached"] = 1
        is_flat, prop, clip, absdiff = self.check_flat_line()
        is_sat, sat_frac = self.check_saturation()
        r.update(flat_proportion=prop, clip_fraction=clip, is_flat=is_flat,
                 raw_sat_fraction=sat_frac, absdiff_norm=absdiff)
        if is_sat:
            r["label"] = (f"Unacceptable – Saturation / Railed Noise "
                          f"({sat_frac:.0%} of raw samples at the rails)")
            return r
        if is_flat:
            why = "Saturation/Clipping" if clip > self.CLIP_THRESH else "Flat Line"
            r["label"] = f"Unacceptable – {why}"
            return r

        # Stage 2 — QRS / HR feasibility
        r["stage_reached"] = 2
        peaks = self._primary_peaks()
        hr_info = self._hr_info(peaks)
        r.update(peaks=peaks, hr_info=hr_info)
        passes, mean_hr, amp_cov, note = self.check_qrs(peaks, hr_info)
        r.update(mean_hr=mean_hr, amp_cov=amp_cov)
        if not passes:
            r["label"] = f"Unacceptable – QRS / HR Check Failed\n({note})"
            return r

        # Stage 3 — beat morphology (iScore), which also tiers the result
        r["stage_reached"] = 3
        iscore, M_x, G_x, beats = self.compute_iscore(peaks, hr_info)
        r.update(iscore=iscore, M_x=M_x, G_x=G_x, beats=beats)

        if iscore < self.ISCORE_L1:
            r["iscore_quality"] = "bad"
            r["label"] = (f"Unacceptable – Low Beat Morphology Consistency\n"
                          f"(iScore = {iscore:.3f} < L1 = {self.ISCORE_L1})")
        elif iscore >= self.ISCORE_L2:
            r["quality"] = "acceptable"
            r["iscore_quality"] = "diag"
            r["label"] = (f"Acceptable – Diagnostic Quality\n"
                          f"(iScore = {iscore:.3f} ≥ L2 = {self.ISCORE_L2})")
        else:
            r["quality"] = "acceptable"
            r["iscore_quality"] = "hr"
            r["label"] = (f"Acceptable – HR Quality\n"
                          f"(L1 ≤ iScore = {iscore:.3f} < L2)")
        return r
