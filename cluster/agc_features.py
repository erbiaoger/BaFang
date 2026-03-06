from math import ceil, floor
from typing import Dict, Optional

import numpy as np
from scipy.signal import correlate, find_peaks, hilbert, resample


_EPS = 1e-8
_PEAK_WINDOW = 256
_TEMPLATE_LEN = 64


def agc_feature_names() -> list[str]:
    return [
        "main_peak_width_50",
        "main_peak_width_25",
        "main_peak_area_ratio",
        "energy_center_span",
        "energy_80_width",
        "peak_left_width",
        "peak_right_width",
        "peak_asymmetry",
        "post_peak_ring_count",
        "post_peak_decay_ratio",
        "pre_peak_energy_ratio",
        "num_major_peaks",
        "num_major_troughs",
        "peak_trough_amplitude_ratio",
        "peak_spacing_mean",
        "peak_spacing_std",
        "corr_to_smallcar_template_max",
        "dtw_to_smallcar_template_min",
        "xcorr_peak_lag",
        "origin_log_rms",
        "origin_envelope_width_50",
        "origin_num_envelope_peaks",
        "origin_energy_gini",
    ]


def feature_index_map() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(agc_feature_names())}


def _normalize_maxabs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / peak if peak > 0 else x


def _vector_unit_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(np.mean(x))
    norm = float(np.linalg.norm(x))
    return x / norm if norm > _EPS else x


def _envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(np.asarray(x, dtype=np.float64)))


def _peak_bounds(envelope: np.ndarray, peak_idx: int, threshold_ratio: float) -> tuple[int, int]:
    threshold = float(np.max(envelope)) * threshold_ratio
    left = peak_idx
    right = peak_idx
    while left > 0 and envelope[left - 1] >= threshold:
        left -= 1
    while right < envelope.size - 1 and envelope[right + 1] >= threshold:
        right += 1
    return left, right


def _energy_gini(x: np.ndarray, n_segments: int = 20) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    seg_len = max(1, x.size // n_segments)
    seg_energies = np.array(
        [float(np.sum(np.square(x[s * seg_len : min((s + 1) * seg_len, x.size)]))) for s in range(n_segments)],
        dtype=np.float64,
    )
    total = float(np.sum(seg_energies))
    if total <= _EPS:
        return 0.0
    seg_sorted = np.sort(seg_energies)
    cumulative = np.cumsum(seg_sorted)
    return float(1.0 - 2.0 * np.sum(cumulative) / (total * n_segments))


def _num_envelope_peaks(x: np.ndarray) -> float:
    env = _envelope(_normalize_maxabs(x))
    peak_val = float(np.max(env))
    if peak_val <= _EPS:
        return 0.0
    smoothed = np.convolve(env, np.ones(min(41, max(3, env.size // 20))) / min(41, max(3, env.size // 20)), mode="same")
    peaks, _ = find_peaks(smoothed, prominence=0.2 * peak_val, distance=max(8, env.size // 60))
    return float(len(peaks))


def _envelope_width_50(x: np.ndarray) -> float:
    env = _envelope(_normalize_maxabs(x))
    if env.size == 0:
        return 0.0
    peak_idx = int(np.argmax(env))
    left, right = _peak_bounds(env, peak_idx, threshold_ratio=0.5)
    return float(right - left + 1)


def _origin_aux_features(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return [
        float(np.log10(rms + _EPS)),
        _envelope_width_50(x),
        _num_envelope_peaks(x),
        _energy_gini(x),
    ]


def _extract_peak_window(x: np.ndarray, window: int = _PEAK_WINDOW) -> np.ndarray:
    x = _normalize_maxabs(x)
    env = _envelope(x)
    if env.size == 0:
        return np.zeros(window, dtype=np.float32)
    peak_idx = int(np.argmax(env))
    half = window // 2
    left = peak_idx - half
    right = left + window
    pad_left = max(0, -left)
    pad_right = max(0, right - x.size)
    src_left = max(0, left)
    src_right = min(x.size, right)
    segment = x[src_left:src_right]
    if pad_left or pad_right:
        segment = np.pad(segment, (pad_left, pad_right), mode="constant")
    if segment.size != window:
        segment = np.resize(segment, window)
    return segment.astype(np.float32, copy=False)


def _resample_signal(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.size == target_len:
        return x.astype(np.float32, copy=False)
    return resample(x.astype(np.float64), target_len).astype(np.float32)


def _banded_dtw(x: np.ndarray, y: np.ndarray, band: int = 6) -> float:
    n = x.size
    m = y.size
    if n == 0 or m == 0:
        return 0.0
    band = max(band, abs(n - m))
    inf = np.float64(1e18)
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        xi = float(x[i - 1])
        for j in range(j_start, j_end + 1):
            cost = abs(xi - float(y[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


def build_smallcar_templates(
    signals_agc: np.ndarray,
    max_templates: int = 1,
    max_samples: int = 30,
    peak_window: int = _PEAK_WINDOW,
    template_len: int = _TEMPLATE_LEN,
) -> np.ndarray:
    signals_agc = np.asarray(signals_agc)
    if signals_agc.ndim == 1:
        signals_agc = signals_agc[np.newaxis, :]
    if signals_agc.shape[0] == 0:
        return np.zeros((0, template_len), dtype=np.float32)

    use_n = min(signals_agc.shape[0], max_samples)
    pick_idx = np.linspace(0, signals_agc.shape[0] - 1, num=use_n, dtype=int)
    windows = np.stack(
        [_vector_unit_norm(_resample_signal(_extract_peak_window(signals_agc[i], peak_window), template_len)) for i in pick_idx],
        axis=0,
    )

    if max_templates <= 1 or windows.shape[0] < max_templates:
        return np.median(windows, axis=0, keepdims=True).astype(np.float32)

    widths = np.array([_envelope_width_50(w) for w in windows], dtype=np.float32)
    order = np.argsort(widths)
    chunks = np.array_split(order, max_templates)
    templates = []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        templates.append(np.median(windows[chunk], axis=0))
    return np.asarray(templates, dtype=np.float32)


def _template_features(x: np.ndarray, templates: Optional[np.ndarray]) -> list[float]:
    if templates is None or len(templates) == 0:
        return [0.0, 0.0, 0.0]

    peak_window = _vector_unit_norm(_resample_signal(_extract_peak_window(x, _PEAK_WINDOW), _TEMPLATE_LEN))
    dtw_window = _resample_signal(_extract_peak_window(x, _PEAK_WINDOW), _TEMPLATE_LEN)
    corr_best = -1.0
    dtw_best = float("inf")
    lag_best = 0.0

    for template in templates:
        tmpl = _vector_unit_norm(template)
        corr = float(np.dot(peak_window, tmpl))
        if corr > corr_best:
            corr_best = corr
            corr_curve = correlate(peak_window, tmpl, mode="full")
            lag_best = float(np.argmax(corr_curve) - (tmpl.size - 1))

        dtw = _banded_dtw(dtw_window, tmpl, band=6)
        if dtw < dtw_best:
            dtw_best = dtw

    return [corr_best, 0.0 if not np.isfinite(dtw_best) else dtw_best, lag_best]


def _single_agc_features(x: np.ndarray) -> list[float]:
    x_norm = _normalize_maxabs(x)
    env = _envelope(x_norm)
    n = x_norm.size
    if n == 0 or env.size == 0:
        return [0.0] * 19

    peak_idx = int(np.argmax(env))
    peak_val = float(np.max(env))
    if peak_val <= _EPS:
        return [0.0] * 19

    left50, right50 = _peak_bounds(env, peak_idx, threshold_ratio=0.5)
    left25, right25 = _peak_bounds(env, peak_idx, threshold_ratio=0.25)
    main_peak_width_50 = float(right50 - left50 + 1)
    main_peak_width_25 = float(right25 - left25 + 1)
    total_env = float(np.sum(env) + _EPS)
    main_peak_area_ratio = float(np.sum(env[left25 : right25 + 1]) / total_env)

    energy = np.square(x_norm.astype(np.float64))
    total_energy = float(np.sum(energy) + _EPS)
    cumulative = np.cumsum(energy) / total_energy
    idx10 = int(np.searchsorted(cumulative, 0.10))
    idx25 = int(np.searchsorted(cumulative, 0.25))
    idx75 = int(np.searchsorted(cumulative, 0.75))
    idx90 = int(np.searchsorted(cumulative, 0.90))
    energy_center = int(np.sum(np.arange(n) * energy) / total_energy)
    energy_center_span = float(abs(energy_center - peak_idx))
    energy_80_width = float(max(0, idx90 - idx10))

    peak_left_width = float(max(0, peak_idx - left50))
    peak_right_width = float(max(0, right50 - peak_idx))
    peak_asymmetry = float((peak_left_width - peak_right_width) / (peak_left_width + peak_right_width + _EPS))

    post_start = min(n, peak_idx + 8)
    post_segment = np.abs(x_norm[post_start:])
    if post_segment.size > 0:
        post_peaks, _ = find_peaks(post_segment, prominence=0.05, distance=max(4, n // 100))
        post_peak_ring_count = float(len(post_peaks))
    else:
        post_peak_ring_count = 0.0

    head_end = min(n, peak_idx + max(16, n // 12))
    tail_start = min(n, peak_idx + max(32, n // 8))
    tail_end = min(n, peak_idx + max(64, n // 4))
    head_energy = float(np.sum(np.abs(x_norm[peak_idx:head_end])) + _EPS)
    tail_energy = float(np.sum(np.abs(x_norm[tail_start:tail_end])))
    post_peak_decay_ratio = float(tail_energy / head_energy)
    pre_peak_energy_ratio = float(np.sum(energy[:peak_idx]) / total_energy)

    prominence = max(0.08, 0.15 * float(np.max(np.abs(x_norm))))
    distance = max(8, n // 80)
    pos_peaks, pos_props = find_peaks(x_norm, prominence=prominence, distance=distance)
    neg_peaks, neg_props = find_peaks(-x_norm, prominence=prominence, distance=distance)
    num_major_peaks = float(len(pos_peaks))
    num_major_troughs = float(len(neg_peaks))

    peak_heights = x_norm[pos_peaks] if len(pos_peaks) else np.array([0.0], dtype=np.float32)
    trough_depths = np.abs(x_norm[neg_peaks]) if len(neg_peaks) else np.array([0.0], dtype=np.float32)
    peak_trough_amplitude_ratio = float((np.mean(peak_heights) + _EPS) / (np.mean(trough_depths) + _EPS))
    peak_spacing_mean = float(np.mean(np.diff(pos_peaks))) if len(pos_peaks) >= 2 else 0.0
    peak_spacing_std = float(np.std(np.diff(pos_peaks))) if len(pos_peaks) >= 2 else 0.0

    template_feats = _template_features(x_norm, None)
    _ = pos_props, neg_props
    return [
        main_peak_width_50,
        main_peak_width_25,
        main_peak_area_ratio,
        energy_center_span,
        energy_80_width,
        peak_left_width,
        peak_right_width,
        peak_asymmetry,
        post_peak_ring_count,
        post_peak_decay_ratio,
        pre_peak_energy_ratio,
        num_major_peaks,
        num_major_troughs,
        peak_trough_amplitude_ratio,
        peak_spacing_mean,
        peak_spacing_std,
        template_feats[0],
        template_feats[1],
        template_feats[2],
    ]


def extract_agc_features(
    signals_agc: np.ndarray,
    signals_origin: Optional[np.ndarray] = None,
    templates: Optional[np.ndarray] = None,
) -> np.ndarray:
    signals_agc = np.asarray(signals_agc)
    if signals_agc.ndim == 1:
        signals_agc = signals_agc[np.newaxis, :]

    if signals_origin is not None:
        signals_origin = np.asarray(signals_origin)
        if signals_origin.ndim == 1:
            signals_origin = signals_origin[np.newaxis, :]
        if signals_origin.shape[0] != signals_agc.shape[0]:
            raise ValueError("signals_origin must align with signals_agc")

    feats = []
    for idx, signal_agc in enumerate(signals_agc):
        base = _single_agc_features(signal_agc)
        if templates is not None and len(templates) > 0:
            template_feats = _template_features(signal_agc, templates)
            base[16:19] = template_feats
        if signals_origin is not None:
            origin_feats = _origin_aux_features(signals_origin[idx])
        else:
            origin_feats = [0.0, 0.0, 0.0, 0.0]
        feats.append(base + origin_feats)
    return np.asarray(feats, dtype=np.float32)


def derive_smallcar_rule_config(
    feature_matrix: np.ndarray,
    positive_mask: np.ndarray,
) -> Dict[str, float]:
    feat_idx = feature_index_map()
    pos = np.asarray(feature_matrix)[np.asarray(positive_mask, dtype=bool)]
    if pos.shape[0] == 0:
        return {
            "corr_min": 0.55,
            "width50_max": 220.0,
            "num_major_peaks_min": 1.0,
            "num_major_peaks_max": 3.0,
            "post_peak_decay_ratio_max": 0.75,
        }

    corr = pos[:, feat_idx["corr_to_smallcar_template_max"]]
    width50 = pos[:, feat_idx["main_peak_width_50"]]
    peak_count = pos[:, feat_idx["num_major_peaks"]]
    decay = pos[:, feat_idx["post_peak_decay_ratio"]]
    return {
        "corr_min": float(max(0.35, np.quantile(corr, 0.25))),
        "width50_max": float(max(24.0, np.quantile(width50, 0.75))),
        "num_major_peaks_min": float(max(1.0, floor(np.quantile(peak_count, 0.10)))),
        "num_major_peaks_max": float(max(2.0, ceil(np.quantile(peak_count, 0.90)))),
        "post_peak_decay_ratio_max": float(min(0.95, np.quantile(decay, 0.75))),
    }


def apply_smallcar_rule(feature_matrix: np.ndarray, config: Dict[str, float]) -> np.ndarray:
    feat_idx = feature_index_map()
    feats = np.asarray(feature_matrix)
    corr = feats[:, feat_idx["corr_to_smallcar_template_max"]]
    width50 = feats[:, feat_idx["main_peak_width_50"]]
    peak_count = feats[:, feat_idx["num_major_peaks"]]
    decay = feats[:, feat_idx["post_peak_decay_ratio"]]
    return (
        (corr >= float(config["corr_min"]))
        & (width50 <= float(config["width50_max"]))
        & (peak_count >= float(config["num_major_peaks_min"]))
        & (peak_count <= float(config["num_major_peaks_max"]))
        & (decay <= float(config["post_peak_decay_ratio_max"]))
    )
