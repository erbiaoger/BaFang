import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis, skew
import pycwt as wavelet


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))


def _spectral_features(x: np.ndarray, eps: float = 1e-12) -> dict:
    n = x.size
    if n == 0:
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 0.0,
            "low_high_energy_ratio": 0.0,
            "dominant_freq": 0.0,
        }

    spec = np.abs(np.fft.rfft(x))
    power = np.square(spec)
    freqs = np.fft.rfftfreq(n, d=1.0)

    total_power = float(np.sum(power) + eps)
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))

    cumulative = np.cumsum(power)
    rolloff_threshold = 0.85 * total_power
    rolloff_idx = int(np.searchsorted(cumulative, rolloff_threshold))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    geom_mean = float(np.exp(np.mean(np.log(power + eps))))
    arith_mean = float(np.mean(power + eps))
    flatness = float(geom_mean / arith_mean)

    split = max(1, len(power) // 4)
    low_energy = float(np.sum(power[:split]))
    high_energy = float(np.sum(power[split:]))
    ratio = float(low_energy / (high_energy + eps))

    dom_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dom_idx])

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "low_high_energy_ratio": ratio,
        "dominant_freq": dominant_freq,
    }


def _stft_band_features(x: np.ndarray, eps: float = 1e-12) -> dict:
    # STFT-based energy ratios and centroid dynamics
    n = x.size
    if n < 8:
        return {
            "stft_low_ratio": 0.0,
            "stft_mid_ratio": 0.0,
            "stft_high_ratio": 0.0,
            "stft_bandwidth": 0.0,
            "stft_centroid_mean": 0.0,
            "stft_centroid_std": 0.0,
        }

    nperseg = min(256, n)
    noverlap = min(nperseg // 2, nperseg - 1)
    f, _, zxx = stft(x, nperseg=nperseg, noverlap=noverlap, boundary=None)
    power = np.abs(zxx) ** 2
    if power.size == 0 or f.size == 0:
        return {
            "stft_low_ratio": 0.0,
            "stft_mid_ratio": 0.0,
            "stft_high_ratio": 0.0,
            "stft_bandwidth": 0.0,
            "stft_centroid_mean": 0.0,
            "stft_centroid_std": 0.0,
        }

    band1 = 0.1 * f.max()
    band2 = 0.3 * f.max()
    low_mask = f <= band1
    mid_mask = (f > band1) & (f <= band2)
    high_mask = f > band2

    total = float(np.sum(power) + eps)
    low = float(np.sum(power[low_mask, :]))
    mid = float(np.sum(power[mid_mask, :]))
    high = float(np.sum(power[high_mask, :]))

    low_ratio = low / total
    mid_ratio = mid / total
    high_ratio = high / total

    power_f = np.sum(power, axis=1)
    total_f = float(np.sum(power_f) + eps)
    centroid = float(np.sum(f * power_f) / total_f)
    bandwidth = float(np.sqrt(np.sum(((f - centroid) ** 2) * power_f) / total_f))

    # centroid over time
    power_t = power + eps
    denom_t = np.sum(power_t, axis=0)
    centroid_t = np.sum(f[:, None] * power_t, axis=0) / denom_t
    centroid_mean = float(np.mean(centroid_t))
    centroid_std = float(np.std(centroid_t))

    return {
        "stft_low_ratio": low_ratio,
        "stft_mid_ratio": mid_ratio,
        "stft_high_ratio": high_ratio,
        "stft_bandwidth": bandwidth,
        "stft_centroid_mean": centroid_mean,
        "stft_centroid_std": centroid_std,
    }


def _wavelet_energy_features(x: np.ndarray, eps: float = 1e-12) -> dict:
    # Morlet wavelet energy ratios across scales (pycwt)
    n = x.size
    if n < 32:
        return {
            "wav_low_ratio": 0.0,
            "wav_mid_ratio": 0.0,
            "wav_high_ratio": 0.0,
        }

    dt = 1.0
    dj = 1 / 12
    s0 = 2 * dt
    J = int(7 / dj)
    wave = wavelet.Morlet(6)
    coeffs, scales, _, _, _, _ = wavelet.cwt(x, dt, dj, s0, J, wave)
    power = np.abs(coeffs) ** 2
    if power.size == 0:
        return {
            "wav_low_ratio": 0.0,
            "wav_mid_ratio": 0.0,
            "wav_high_ratio": 0.0,
        }

    energy = np.sum(power, axis=1)
    total = float(np.sum(energy) + eps)
    # Split scales into low/mid/high by thirds
    n_scales = energy.shape[0]
    a = max(1, n_scales // 3)
    low = float(np.sum(energy[:a]))
    mid = float(np.sum(energy[a:2 * a]))
    high = float(np.sum(energy[2 * a:]))

    return {
        "wav_low_ratio": low / total,
        "wav_mid_ratio": mid / total,
        "wav_high_ratio": high / total,
    }


def extract_features(signals: np.ndarray) -> np.ndarray:
    feats = []
    for x in signals:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            feats.append([0.0] * 23)
            continue
        mean = float(np.mean(x))
        std = float(np.std(x))
        rms = _rms(x)
        ptp = float(np.max(x) - np.min(x))
        sk = float(skew(x)) if x.size > 2 else 0.0
        ku = float(kurtosis(x)) if x.size > 3 else 0.0
        zcr = _zero_crossing_rate(x)
        energy = float(np.sum(np.square(x)))

        spec = _spectral_features(x)
        stft_feat = _stft_band_features(x)
        wav_feat = _wavelet_energy_features(x)

        feat = [
            mean,
            std,
            rms,
            ptp,
            sk,
            ku,
            zcr,
            energy,
            spec["spectral_centroid"],
            spec["spectral_bandwidth"],
            spec["spectral_rolloff"],
            spec["spectral_flatness"],
            spec["low_high_energy_ratio"],
            spec["dominant_freq"],
            stft_feat["stft_low_ratio"],
            stft_feat["stft_mid_ratio"],
            stft_feat["stft_high_ratio"],
            stft_feat["stft_bandwidth"],
            stft_feat["stft_centroid_mean"],
            stft_feat["stft_centroid_std"],
            wav_feat["wav_low_ratio"],
            wav_feat["wav_mid_ratio"],
            wav_feat["wav_high_ratio"],
        ]
        feats.append(feat)
    return np.asarray(feats, dtype=np.float32)


def feature_names() -> list[str]:
    return [
        "mean",
        "std",
        "rms",
        "peak_to_peak",
        "skew",
        "kurtosis",
        "zero_crossing_rate",
        "signal_energy",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "low_high_energy_ratio",
        "dominant_freq",
        "stft_low_ratio",
        "stft_mid_ratio",
        "stft_high_ratio",
        "stft_bandwidth",
        "stft_centroid_mean",
        "stft_centroid_std",
        "wav_low_ratio",
        "wav_mid_ratio",
        "wav_high_ratio",
    ]
