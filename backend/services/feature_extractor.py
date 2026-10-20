import numpy as np
from scipy import signal
from schemas import AcousticFeatures


def extract_acoustic_features(audio: np.ndarray, sample_rate: int) -> AcousticFeatures:
    """
    Extracts core acoustic and spectral features from audio samples for deepfake
    and synthetic voice analysis.
    """
    duration = float(len(audio) / sample_rate)

    # 1. Zero Crossing Rate (ZCR)
    sign_changes = np.abs(np.diff(np.sign(audio)))
    zcr = float(np.mean(sign_changes) / 2.0)

    # 2. Root Mean Square (RMS) Energy
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))

    # 3. Spectral Features via Short-Time Fourier Transform (STFT)
    nperseg = min(1024, len(audio))
    if nperseg < 64:
        nperseg = len(audio)

    f, _, zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    mag_spec = np.abs(zxx)
    power_spec = mag_spec ** 2

    # Spectral Centroid (center of mass of the spectrum)
    freq_grid = f[:, np.newaxis]
    total_energy_per_frame = np.sum(mag_spec, axis=0) + 1e-10
    frame_centroids = np.sum(freq_grid * mag_spec, axis=0) / total_energy_per_frame
    spectral_centroid = float(np.mean(frame_centroids))

    # Spectral Rolloff (85% energy threshold)
    cumsum_power = np.cumsum(power_spec, axis=0)
    total_frame_power = cumsum_power[-1, :] + 1e-10
    rolloff_mask = cumsum_power >= (0.85 * total_frame_power)
    rolloff_indices = np.argmax(rolloff_mask, axis=0)
    spectral_rolloff = float(np.mean(f[rolloff_indices]))

    # Spectral Flatness (Wiener entropy = geometric mean / arithmetic mean)
    eps = 1e-10
    log_power = np.log(power_spec + eps)
    geom_mean = np.exp(np.mean(log_power, axis=0))
    arith_mean = np.mean(power_spec, axis=0) + eps
    spectral_flatness = float(np.mean(geom_mean / arith_mean))

    # 4. Fundamental Frequency (F0), Jitter, and Shimmer (Pitch Tracking)
    f0_mean, jitter, shimmer = _estimate_pitch_and_perturbation(audio, sample_rate)

    return AcousticFeatures(
        sample_rate=sample_rate,
        duration_seconds=round(duration, 3),
        channels=1,
        zero_crossing_rate=round(zcr, 4),
        rms_energy=round(rms_energy, 4),
        spectral_centroid_hz=round(spectral_centroid, 2),
        spectral_rolloff_hz=round(spectral_rolloff, 2),
        spectral_flatness=round(spectral_flatness, 4),
        fundamental_frequency_hz=round(f0_mean, 2) if f0_mean is not None else None,
        pitch_jitter=round(jitter, 4) if jitter is not None else None,
        shimmer=round(shimmer, 4) if shimmer is not None else None,
    )


def _estimate_pitch_and_perturbation(
    audio: np.ndarray, sample_rate: int
) -> tuple[float | None, float | None, float | None]:
    """
    Estimates F0 (pitch), Jitter (frequency perturbation), and Shimmer
    (amplitude perturbation) using frame-based normalized autocorrelation.
    """
    frame_len = int(0.04 * sample_rate)  # 40ms frame
    hop_len = int(0.02 * sample_rate)    # 20ms hop
    min_lag = int(sample_rate / 500)     # Max 500 Hz human pitch
    max_lag = int(sample_rate / 70)      # Min 70 Hz human pitch

    if len(audio) < frame_len or max_lag >= frame_len:
        return None, None, None

    f0_values = []
    amplitudes = []

    num_frames = (len(audio) - frame_len) // hop_len
    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        frame = frame - np.mean(frame)
        frame_rms = np.sqrt(np.mean(frame ** 2))

        # Ignore silence/near-silent frames
        if frame_rms < 0.01:
            continue

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(frame) - 1 :]
        r0 = autocorr[0] + 1e-10

        norm_autocorr = autocorr[min_lag:max_lag] / r0
        if len(norm_autocorr) == 0:
            continue

        peak_idx = np.argmax(norm_autocorr)
        peak_val = norm_autocorr[peak_idx]

        # Valid pitch candidate peak threshold
        if peak_val > 0.35:
            lag = min_lag + peak_idx
            f0 = sample_rate / lag
            f0_values.append(f0)
            amplitudes.append(frame_rms)

    if len(f0_values) < 5:
        return None, None, None

    f0_arr = np.array(f0_values)
    amp_arr = np.array(amplitudes)

    f0_mean = float(np.mean(f0_arr))

    # Jitter: cycle-to-cycle F0 relative variation
    f0_diffs = np.abs(np.diff(f0_arr))
    jitter = float(np.mean(f0_diffs) / (f0_mean + 1e-10))

    # Shimmer: cycle-to-cycle amplitude relative variation
    amp_diffs = np.abs(np.diff(amp_arr))
    shimmer = float(np.mean(amp_diffs) / (np.mean(amp_arr) + 1e-10))

    return f0_mean, jitter, shimmer
