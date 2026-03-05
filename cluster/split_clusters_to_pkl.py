import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from diffusion.diffusion_dataset import (  # noqa: E402
    build_signal_matrix,
    load_records_from_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split clustered samples into per-class grouped pkl files")
    parser.add_argument("--pkl_dir", required=True, help="Directory containing grouped .pkl files")
    parser.add_argument("--clusters_csv", required=True, help="clusters_raw.csv (sample_id -> cluster_id)")
    parser.add_argument("--out_dir", required=True, help="Output directory for class pkls")
    parser.add_argument("--data_info", default=None, help="Optional data_info.json for length_mode/target_length")
    return parser.parse_args()


def _read_clusters(path: Path) -> Dict[int, int]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["sample_id"])] = int(row["cluster_id"])
    return mapping


def _load_data_info(path: Path) -> Tuple[int | None, str | None]:
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    target_length = info.get("target_length")
    length_mode = info.get("length_mode")
    return target_length, length_mode


def _init_grouped() -> Dict:
    return defaultdict(lambda: {"value": [], "station": [], "sta_name": [], "time": [], "veh_id": None, "fs": None})


def main() -> None:
    args = parse_args()
    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters = _read_clusters(Path(args.clusters_csv))
    max_id = max(clusters.keys()) if clusters else -1

    data_info_path = Path(args.data_info) if args.data_info else None
    if data_info_path is None:
        default_info = Path(args.clusters_csv).parent / "data_info.json"
        data_info_path = default_info
    target_len, length_mode = _load_data_info(data_info_path)
    length_mode = length_mode or "crop"

    records = load_records_from_dir(str(pkl_dir))
    signals, metas, info = build_signal_matrix(
        records,
        sample_length=target_len,
        length_mode=length_mode,
    )

    if max_id >= len(metas):
        raise ValueError(f"clusters_csv sample_id max={max_id} exceeds meta length={len(metas)}")
    if len(clusters) != len(metas):
        raise ValueError(f"clusters_csv rows={len(clusters)} mismatch meta length={len(metas)}")

    class_grouped = {0: _init_grouped(), 1: _init_grouped(), 2: _init_grouped()}
    class_counts = {0: 0, 1: 0, 2: 0}

    for i, meta in enumerate(metas):
        label = clusters.get(i)
        if label is None:
            raise ValueError(f"missing label for sample_id={i}")
        if label not in class_grouped:
            continue
        grp = class_grouped[label]
        veh_id = meta.get("veh_id")
        g = grp[veh_id]
        g["veh_id"] = veh_id
        g["value"].append(signals[i].astype(np.float32, copy=False))
        g["station"].append(meta.get("station", -1))
        g["sta_name"].append(meta.get("sta_name", ""))
        g["time"].append(meta.get("time", -1))
        class_counts[label] += 1

    summary = {
        "counts": {str(k): int(v) for k, v in class_counts.items()},
        "num_total": int(len(metas)),
        "target_length": int(info["target_length"]),
        "length_mode": info["length_mode"],
        "clusters_csv": str(Path(args.clusters_csv)),
        "pkl_dir": str(pkl_dir),
        "data_info": str(data_info_path),
    }

    for label, grouped in class_grouped.items():
        class_dir = out_dir / f"class_{label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"class_{label}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(dict(grouped), f)

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved class pkls to: {out_dir}")


if __name__ == "__main__":
    main()
