import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from diffusion.diffusion_dataset import build_signal_matrix, flatten_grouped_records, load_grouped_pkl, load_records_from_dir  # noqa: E402


META_FIELDS = [
    "sample_id",
    "veh_id",
    "station",
    "sta_name",
    "time",
    "source_origin",
    "source_agc",
    "agc_matched",
    "agc_sample_index",
    "cluster_id_old",
    "label_manual",
    "split",
]


def load_records(path: Path) -> List[Dict]:
    if path.is_dir():
        return load_records_from_dir(str(path))
    if path.is_file() and path.suffix.lower() == ".pkl":
        grouped = load_grouped_pkl(path)
        return flatten_grouped_records(grouped, path)
    raise FileNotFoundError(f"not found or unsupported: {path}")


def read_clusters_csv(path: Optional[Path]) -> Dict[int, int]:
    if path is None:
        return {}
    mapping: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["sample_id"])] = int(row["cluster_id"])
    return mapping


def read_meta_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_meta_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _meta_key(meta: Dict, with_source: bool = False) -> Tuple:
    key = (meta.get("veh_id"), meta.get("station"), meta.get("time"))
    if with_source:
        key = key + (meta.get("source"),)
    return key


def build_signal_index(
    records: List[Dict],
    sample_length: int,
    length_mode: str = "crop",
) -> Dict:
    signals, meta, info = build_signal_matrix(records, sample_length=sample_length, length_mode=length_mode)
    key3_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    key4_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    occ_by_index: List[int] = [0] * len(meta)
    for idx, m in enumerate(meta):
        key3 = _meta_key(m, with_source=False)
        key4 = _meta_key(m, with_source=True)
        occ_by_index[idx] = len(key3_to_indices[key3])
        key3_to_indices[key3].append(idx)
        key4_to_indices[key4].append(idx)
    return {
        "signals": signals.astype(np.float32, copy=False),
        "meta": meta,
        "info": info,
        "key3_to_indices": dict(key3_to_indices),
        "key4_to_indices": dict(key4_to_indices),
        "occ_by_index": occ_by_index,
    }


def build_paired_dataset(
    origin_path: Path,
    agc_path: Path,
    sample_length: int,
    length_mode: str = "crop",
    clusters_csv: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict]:
    cluster_map = read_clusters_csv(clusters_csv)

    origin_records = load_records(origin_path)
    agc_records = load_records(agc_path)
    origin_index = build_signal_index(origin_records, sample_length=sample_length, length_mode=length_mode)
    agc_index = build_signal_index(agc_records, sample_length=sample_length, length_mode=length_mode)

    signals_origin = origin_index["signals"]
    signals_agc = np.zeros_like(signals_origin, dtype=np.float32)
    meta_rows: List[Dict] = []

    total = len(signals_origin)
    matched = 0
    missing = 0
    per_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "matched": 0, "missing": 0})

    for sample_id, origin_meta in enumerate(origin_index["meta"]):
        key3 = _meta_key(origin_meta, with_source=False)
        occ = origin_index["occ_by_index"][sample_id]
        agc_candidates = agc_index["key3_to_indices"].get(key3, [])
        agc_idx = agc_candidates[occ] if occ < len(agc_candidates) else None

        source_origin = str(origin_meta["source"])
        per_source[source_origin]["total"] += 1

        if agc_idx is None:
            missing += 1
            per_source[source_origin]["missing"] += 1
            source_agc = ""
            agc_matched = 0
        else:
            matched += 1
            per_source[source_origin]["matched"] += 1
            signals_agc[sample_id] = agc_index["signals"][agc_idx]
            source_agc = str(agc_index["meta"][agc_idx]["source"])
            agc_matched = 1

        meta_rows.append(
            {
                "sample_id": sample_id,
                "veh_id": origin_meta.get("veh_id", ""),
                "station": origin_meta.get("station", ""),
                "sta_name": origin_meta.get("sta_name", ""),
                "time": origin_meta.get("time", ""),
                "source_origin": source_origin,
                "source_agc": source_agc,
                "agc_matched": agc_matched,
                "agc_sample_index": "" if agc_idx is None else agc_idx,
                "cluster_id_old": "" if sample_id not in cluster_map else cluster_map[sample_id],
                "label_manual": "",
                "split": "",
            }
        )

    stats = {
        "num_samples": total,
        "num_matched": matched,
        "num_missing": missing,
        "match_rate": float(matched / total) if total else 0.0,
        "sample_length": sample_length,
        "length_mode": length_mode,
        "origin_path": str(origin_path),
        "agc_path": str(agc_path),
        "per_source": dict(per_source),
    }
    return signals_origin, signals_agc, meta_rows, stats


def save_paired_dataset(
    dataset_npz: Path,
    meta_csv: Path,
    signals_origin: np.ndarray,
    signals_agc: np.ndarray,
    meta_rows: List[Dict],
    stats: Dict,
) -> None:
    dataset_npz.parent.mkdir(parents=True, exist_ok=True)
    origin_npy = dataset_npz.with_name(f"{dataset_npz.stem}_signals_origin.npy")
    agc_npy = dataset_npz.with_name(f"{dataset_npz.stem}_signals_agc.npy")
    stats_json = dataset_npz.with_name(f"{dataset_npz.stem}_stats.json")

    np.save(origin_npy, np.asarray(signals_origin, dtype=np.float32))
    np.save(agc_npy, np.asarray(signals_agc, dtype=np.float32))
    np.savez(
        dataset_npz,
        signals_origin_path=np.asarray(origin_npy.name),
        signals_agc_path=np.asarray(agc_npy.name),
        num_samples=np.asarray(stats.get("num_samples", 0), dtype=np.int64),
        sample_length=np.asarray(stats.get("sample_length", 0), dtype=np.int64),
        match_rate=np.asarray(stats.get("match_rate", 0.0), dtype=np.float64),
    )
    write_meta_csv(meta_csv, meta_rows)
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_paired_dataset(
    dataset_npz: Path,
    meta_csv: Path,
    mmap_mode: Optional[str] = "r",
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict]:
    manifest = np.load(dataset_npz, allow_pickle=True)
    dataset_dir = dataset_npz.parent
    origin_npy = dataset_dir / str(manifest["signals_origin_path"].item())
    agc_npy = dataset_dir / str(manifest["signals_agc_path"].item())
    signals_origin = np.load(origin_npy, mmap_mode=mmap_mode)
    signals_agc = np.load(agc_npy, mmap_mode=mmap_mode)
    meta_rows = read_meta_csv(meta_csv)
    stats = {
        "num_samples": int(manifest["num_samples"].item()),
        "sample_length": int(manifest["sample_length"].item()),
        "match_rate": float(manifest["match_rate"].item()),
        "signals_origin_path": str(origin_npy),
        "signals_agc_path": str(agc_npy),
    }
    return signals_origin, signals_agc, meta_rows, stats
