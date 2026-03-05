import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def collect_pkl_files(pkl_dir: str) -> List[Path]:
    root = Path(pkl_dir)
    if not root.exists():
        raise FileNotFoundError(f"pkl_dir not found: {root}")
    files = sorted(root.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no .pkl files found under: {root}")
    return files


def load_grouped_pkl(pkl_path: Path) -> Dict:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported pkl object type at {pkl_path}: {type(obj)}")
    return obj


def flatten_grouped_records(grouped: Dict, source_path: Path) -> List[Dict]:
    records: List[Dict] = []
    for veh_id, group in grouped.items():
        values = group.get("value", [])
        stations = group.get("station", [])
        sta_names = group.get("sta_name", [])
        times = group.get("time", [])

        n = len(values)
        if n == 0:
            continue

        for i in range(n):
            signal = np.asarray(values[i], dtype=np.float32).reshape(-1)
            if signal.size == 0:
                continue
            record = {
                "signal": signal,
                "veh_id": int(veh_id) if str(veh_id).isdigit() else veh_id,
                "station": int(stations[i]) if i < len(stations) else -1,
                "sta_name": str(sta_names[i]) if i < len(sta_names) else "",
                "time": int(times[i]) if i < len(times) else -1,
                "source": str(source_path),
            }
            records.append(record)
    return records


def load_records_from_dir(pkl_dir: str) -> List[Dict]:
    records: List[Dict] = []
    for p in collect_pkl_files(pkl_dir):
        grouped = load_grouped_pkl(p)
        records.extend(flatten_grouped_records(grouped, p))
    if not records:
        raise ValueError(f"no valid signal records found in: {pkl_dir}")
    return records


def _infer_target_length(records: Sequence[Dict]) -> int:
    lengths = [int(r["signal"].shape[0]) for r in records]
    cnt = Counter(lengths)
    return cnt.most_common(1)[0][0]


def _adjust_length(signal: np.ndarray, target_len: int, mode: str) -> Optional[np.ndarray]:
    n = signal.shape[0]
    if n == target_len:
        return signal
    if mode == "filter":
        return None
    if mode == "crop":
        if n > target_len:
            return signal[:target_len]
        padded = np.zeros(target_len, dtype=signal.dtype)
        padded[:n] = signal
        return padded
    if mode == "pad":
        if n > target_len:
            return signal[:target_len]
        padded = np.zeros(target_len, dtype=signal.dtype)
        padded[:n] = signal
        return padded
    raise ValueError(f"unsupported length mode: {mode}")


def build_signal_matrix(
    records: Sequence[Dict],
    sample_length: Optional[int] = None,
    length_mode: str = "filter",
) -> Tuple[np.ndarray, List[Dict], Dict]:
    target_len = int(sample_length) if sample_length is not None else _infer_target_length(records)

    used_meta: List[Dict] = []
    signals: List[np.ndarray] = []
    dropped = 0

    for r in records:
        sig = _adjust_length(r["signal"], target_len, length_mode)
        if sig is None:
            dropped += 1
            continue
        signals.append(sig.astype(np.float32, copy=False))
        used_meta.append(
            {
                "veh_id": r["veh_id"],
                "station": r["station"],
                "sta_name": r["sta_name"],
                "time": r["time"],
                "source": r["source"],
            }
        )

    if not signals:
        raise ValueError("all records were dropped after length filtering/adjustment")

    matrix = np.stack(signals, axis=0).astype(np.float32)
    info = {
        "target_length": target_len,
        "num_total_records": len(records),
        "num_used_records": len(used_meta),
        "num_dropped_records": dropped,
        "length_mode": length_mode,
    }
    return matrix, used_meta, info


def normalize_signals(
    signals: np.ndarray,
    mode: str = "per_sample",
    eps: float = 1e-6,
    stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    if mode not in {"per_sample", "global"}:
        raise ValueError(f"unsupported normalization mode: {mode}")

    x = signals.astype(np.float32)

    if mode == "per_sample":
        scale = np.max(np.abs(x), axis=1, keepdims=True)
        scale = np.maximum(scale, eps)
        x_norm = x / scale
        return x_norm, {"mode": "per_sample", "eps": eps}

    if stats is None:
        mean = float(np.mean(x))
        std = float(np.std(x))
        std = max(std, eps)
        stats = {"mode": "global", "mean": mean, "std": std, "eps": eps}
    mean = float(stats["mean"])
    std = float(stats["std"])
    x_norm = (x - mean) / std
    return x_norm, stats


def denormalize_signals(signals: np.ndarray, stats: Dict) -> np.ndarray:
    mode = stats.get("mode")
    if mode == "per_sample":
        return signals
    if mode == "global":
        return (signals * float(stats["std"])) + float(stats["mean"])
    raise ValueError(f"unsupported stats mode: {mode}")


def save_stats_json(path: Path, stats: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_stats_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
