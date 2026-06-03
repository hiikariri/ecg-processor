import time
import os
import shutil
import board, busio
import digitalio
import numpy as np
import threading
import csv
import subprocess
import pywt
from scipy.signal import filtfilt, find_peaks
from scipy.interpolate import interp1d
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_mcp3xxx.mcp3008 as MCP
from scipy.signal import butter
from .config import *
from .utils import calculate_psd_manual, calculate_mpf

class ECGProcessor:
    """ECG data acquisition and processing class integrated with sensor module"""
    
    def __init__(self, sampling_freq=ECG_SAMPLING_FREQ, duration=ECG_DURATION):
        self.sampling_freq = sampling_freq
        self.sampling_interval = 1.0 / sampling_freq
        self.duration = duration
        self.use_c_backend = ECG_BACKEND in ("c", "auto")
        self.spi = None
        self.cs = None
        self.mcp = None
        self.chan = None
        
        self.ecg_data = []
        self.time_data = []
        self.filtered_ecg_data = []
        
        self.is_recording = False
        self.start_time = None
        
        self.hardware_initialized = False
        if not self.use_c_backend:
            self._init_python_hardware()

    def _init_python_hardware(self):
        if self.hardware_initialized:
            return True
        try:
            self.spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
            self.cs = digitalio.DigitalInOut(board.CE0)
            self.mcp = MCP.MCP3008(self.spi, self.cs)
            self.chan = AnalogIn(self.mcp, ECG_MCP3008_CHANNEL)
            self.hardware_initialized = True
            return True
        except Exception as e:
            print(f"[ECG] Hardware initialization failed: {e}")
            self.hardware_initialized = False
            return False

    def _ensure_c_binary(self, status_callback=None):
        def update_status(message):
            if status_callback:
                status_callback(message)
            print(f"[ECG-C] {message}")

        if ECG_C_BINARY_PATH.exists() and ECG_C_BINARY_PATH.is_file():
            if ECG_C_SOURCE_PATH.exists() and ECG_C_SOURCE_PATH.stat().st_mtime <= ECG_C_BINARY_PATH.stat().st_mtime:
                return True
        else:
            if not ECG_C_SOURCE_PATH.exists():
                update_status(f"C source not found: {ECG_C_SOURCE_PATH}")
                return False

        if not shutil.which("gcc"):
            update_status("gcc not available; cannot build native ECG collector")
            return False

        ECG_C_BINARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_cmd = [
            "gcc", "-O2", "-std=c11", "-Wall", "-Wextra",
            str(ECG_C_SOURCE_PATH), "-o", str(ECG_C_BINARY_PATH),
            "-lgpiod", "-lrt",
        ]

        update_status("Building native ECG collector...")
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            update_status(f"Native ECG build failed: {stderr}")
            return False

        ECG_C_BINARY_PATH.chmod(0o755)
        update_status("Native ECG collector is ready")
        return True

    def _record_with_c_backend(self, status_callback=None):
        def update_status(message):
            if status_callback:
                status_callback(message)
            print(f"[ECG-C] {message}")

        if not self._ensure_c_binary(status_callback=status_callback):
            return False

        run_cmd = [
            str(ECG_C_BINARY_PATH),
            "--output", str(ECG_C_RAW_OUTPUT_PATH),
            "--duration", str(int(self.duration)),
            "--rate", str(int(self.sampling_freq)),
            "--channel", str(int(ECG_ADC_CHANNEL)),
            "--spi-device", ECG_SPI_DEVICE,
            "--spi-speed", str(int(ECG_SPI_SPEED_HZ)),
            "--cs-gpio", str(int(ECG_CS_GPIO)),
        ]

        update_status(f"Recording ECG via native C ({self.duration}s @ {self.sampling_freq}Hz)...")
        result = subprocess.run(
            run_cmd, capture_output=True, text=True,
            timeout=max(15, int(self.duration) + 20),
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            update_status(f"Native ECG run failed: {stderr}")
            return False

        if not ECG_C_RAW_OUTPUT_PATH.exists():
            update_status("Native ECG output file not found")
            return False

        try:
            raw = np.loadtxt(ECG_C_RAW_OUTPUT_PATH, delimiter=",", skiprows=1)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[1] < 2:
                update_status("Native ECG output format invalid")
                return False

            times = raw[:, 0]
            adc_values = raw[:, 1]
            voltages = (adc_values / 1023.0) * 3.3

            self.time_data = times.tolist()
            self.ecg_data = voltages.tolist()
            update_status(f"Native ECG samples loaded: {len(self.ecg_data)}")
            return len(self.ecg_data) > 0
        except Exception as e:
            update_status(f"Failed to parse native ECG output: {e}")
            return False
    
    def get_ecg_value(self):
        if not self.hardware_initialized:
            return 0.0
        try:
            return self.chan.voltage
        except Exception as e:
            print(f"[ECG] Reading error: {e}")
            return 0.0
    
    def bandpass_filter(self, data, lowcut=ECG_FILTER_LOWCUT, highcut=ECG_FILTER_HIGHCUT, fs=None, order=4):
        if fs is None:
            fs = self.sampling_freq
        data = np.array(data)
        if data.size < order + 1:
            return data
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if low <= 0: low = 0.001
        if high >= 1: high = 0.999
        
        try:
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)
        except Exception as e:
            print(f"[ECG] Bandpass filter error: {e}")
            return data
    
    def apply_ecg_filters(self, data):
        data = np.array(data)
        if data.size < 20:
            return data
        try:
            return self.bandpass_filter(data)
        except Exception as e:
            print(f"[ECG] Filter pipeline error: {e}")
            return data
    
    def calculate_heart_rate(self, ecg_data, fs=None):
        if fs is None:
            fs = self.sampling_freq
        if len(ecg_data) < fs:
            return 0, 0
        
        try:
            wavelet = 'db4'
            level = 7
            coeffs = pywt.wavedec(ecg_data, wavelet, level=level)
            coeffs_denoised = [np.zeros_like(c) if i < 3 or i > 5 else c for i, c in enumerate(coeffs)]
            qrs_reconstructed = pywt.waverec(coeffs_denoised, wavelet)
            qrs_reconstructed = qrs_reconstructed[:len(ecg_data)]

            min_distance = int(0.25 * fs) 
            threshold = 0.3 * np.max(qrs_reconstructed)

            peaks, _ = find_peaks(
                qrs_reconstructed,
                distance=min_distance,
                height=threshold,
                prominence=0.5*np.max(qrs_reconstructed) 
            )
            
            if len(peaks) < 2:
                return 0, 0
            
            rr_intervals = np.diff(peaks) / fs
            valid_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
            if len(valid_intervals) == 0:
                return 0, 0
            
            avg_rr_interval = np.mean(valid_intervals)
            heart_rate = 60.0 / avg_rr_interval
            
            try:
                rr_times = np.cumsum(rr_intervals)
                rr_times = np.insert(rr_times, 0, 0)
                Fs_RR = round(1 / avg_rr_interval, 2) if avg_rr_interval > 0 else 1
                t_uniform = np.arange(0, rr_times[-2], 1 / Fs_RR)
                
                if len(t_uniform) > 1 and len(rr_times) > 2:
                    rr_interp_func = interp1d(rr_times[:-1], rr_intervals, kind='cubic')
                    rr_interp = rr_interp_func(t_uniform)
                    
                    freqs, psd = calculate_psd_manual(rr_interp, Fs_RR, window_size=32, overlap=16, pad_to=256)
                    mpf = calculate_mpf(freqs, psd)
                    respiratory_rate = mpf * 60
                else:
                    respiratory_rate = "error" 
            except:
                respiratory_rate = "error"
            
            return heart_rate, respiratory_rate
            
        except Exception as e:
            print(f"[ECG] Heart rate calculation error: {e}")
            return 0, 0
    
    def data_acquisition_thread(self, status_callback=None):
        def update_status(message):
            if status_callback:
                status_callback(message)
            print(f"[ECG] {message}")
        
        if not self.hardware_initialized and not self.use_c_backend:
            update_status("ECG hardware not initialized")
            return
        
        self.start_time = time.perf_counter()
        update_status(f"Started recording ECG for {self.duration} seconds...")
        
        used_c_backend = False
        if self.use_c_backend:
            used_c_backend = self._record_with_c_backend(status_callback=update_status)
            if not used_c_backend:
                update_status("Native ECG backend failed, falling back to Python")

        if not used_c_backend:
            if not self.hardware_initialized and not self._init_python_hardware():
                update_status("ECG fallback unavailable: Python SPI hardware not initialized")
                self.is_recording = False
                return

            sample_count = 0
            next_sample_time = self.start_time

            while self.is_recording and (time.perf_counter() - self.start_time) < self.duration:
                current_time = time.perf_counter()
                if current_time >= next_sample_time:
                    ecg_value = self.get_ecg_value()
                    relative_time = current_time - self.start_time
                    self.ecg_data.append(ecg_value)
                    self.time_data.append(relative_time)

                    sample_count += 1
                    next_sample_time += self.sampling_interval
        
        self.is_recording = False
        
        if len(self.ecg_data) > 20:
            update_status("Applying ECG filters...")
            try:
                self.filtered_ecg_data = self.apply_ecg_filters(self.ecg_data)
                update_status("ECG filtering completed")
            except Exception as e:
                update_status(f"ECG filtering failed: {e}")
                self.filtered_ecg_data = self.ecg_data.copy()
        else:
            self.filtered_ecg_data = self.ecg_data.copy()
        
        trim_seconds = 5
        n_remove = self.sampling_freq * trim_seconds
        if len(self.ecg_data) > n_remove:
            self.ecg_data = self.ecg_data[n_remove:]
            if len(self.time_data) > n_remove:
                self.time_data = self.time_data[n_remove:]
        
        update_status(f"ECG recording completed: {len(self.ecg_data)} samples")
    
    def start_recording(self, status_callback=None):
        if self.is_recording:
            if status_callback: status_callback("ECG recording already in progress")
            return False
        if not self.hardware_initialized and not self.use_c_backend:
            if status_callback: status_callback("ECG hardware not available")
            return False
        
        self.ecg_data.clear()
        self.time_data.clear()
        self.filtered_ecg_data.clear()
        self.is_recording = True
        
        acquisition_thread = threading.Thread(target=self.data_acquisition_thread, args=(status_callback,), daemon=True)
        acquisition_thread.start()
        return True
    
    def stop_recording(self):
        self.is_recording = False
    
    def get_results(self):
        if not self.ecg_data:
            return None
        try:
            data_to_process = self.filtered_ecg_data if len(self.filtered_ecg_data) > 0 else self.ecg_data
            heart_rate, respiratory_rate = self.calculate_heart_rate(data_to_process)
            signal_quality = "Good" if len(self.filtered_ecg_data) > 0 and np.std(self.filtered_ecg_data) > 0.01 else "Poor"
            return {
                "heart_rate": heart_rate,
                "respiratory_rate": respiratory_rate,
                "signal_quality": signal_quality,
                "duration": len(self.time_data) / self.sampling_freq if self.time_data else 0,
                "samples_collected": len(self.ecg_data)
            }
        except Exception as e:
            print(f"[ECG] Results calculation error: {e}")
            return None
    
    def save_data(self, filename_prefix="ecg_data"):
        if not self.ecg_data:
            return None
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_filename = str(ECG_OUTPUT_PATH)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Raw_Voltage (V)", "Filtered_Voltage (V)"])
                for i, (t, v) in enumerate(zip(self.time_data, self.ecg_data)):
                    filtered_v = self.filtered_ecg_data[i] if i < len(self.filtered_ecg_data) else v
                    writer.writerow([t, v, filtered_v])
            return csv_filename
        except Exception as e:
            print(f"[ECG] Error saving data: {e}")
            return None