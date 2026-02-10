import numpy as np
from scipy.stats import kurtosis, skew


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


def extract_features(signals: np.ndarray) -> np.ndarray:
    feats = []
    for x in signals:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            feats.append([0.0] * 14)
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
    ]
