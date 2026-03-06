import argparse
import csv
import random
import sys
from pathlib import Path


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from cluster.agc_dataset import read_meta_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample manual review candidates for vehicle labeling")
    parser.add_argument("--meta_csv", required=True, help="Metadata CSV from build_agc_training_set.py")
    parser.add_argument("--out_csv", required=True, help="Output candidate CSV for manual labeling")
    parser.add_argument("--cluster1_small", type=int, default=50, help="Samples from old cluster 1")
    parser.add_argument("--cluster0_other", type=int, default=60, help="Samples from old cluster 0")
    parser.add_argument("--cluster2_other", type=int, default=60, help="Samples from old cluster 2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _pick(rows, size, rng):
    if not rows:
        return []
    if len(rows) <= size:
        return list(rows)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:size]]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = read_meta_csv(Path(args.meta_csv))
    rows = [row for row in rows if int(row.get("agc_matched", 0) or 0) == 1]

    by_cluster = {"0": [], "1": [], "2": []}
    for row in rows:
        cluster_id = str(row.get("cluster_id_old", ""))
        if cluster_id in by_cluster:
            by_cluster[cluster_id].append(row)

    selected = []
    for row in _pick(by_cluster["1"], args.cluster1_small, rng):
        selected.append((row, "cluster1_seed", "small"))
    for row in _pick(by_cluster["0"], args.cluster0_other, rng):
        selected.append((row, "cluster0_mix", ""))
    for row in _pick(by_cluster["2"], args.cluster2_other, rng):
        selected.append((row, "cluster2_mix", ""))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "reviewer",
        "note",
        "suggested_label",
        "priority_group",
        "cluster_id_old",
        "veh_id",
        "station",
        "sta_name",
        "time",
        "source_origin",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, priority_group, suggested_label in selected:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "label": "",
                    "reviewer": "",
                    "note": "",
                    "suggested_label": suggested_label,
                    "priority_group": priority_group,
                    "cluster_id_old": row.get("cluster_id_old", ""),
                    "veh_id": row.get("veh_id", ""),
                    "station": row.get("station", ""),
                    "sta_name": row.get("sta_name", ""),
                    "time": row.get("time", ""),
                    "source_origin": row.get("source_origin", ""),
                }
            )

    print(f"saved manual review candidates: {out_path}")
    counts = {"cluster1_seed": 0, "cluster0_mix": 0, "cluster2_mix": 0}
    for _, priority_group, _ in selected:
        counts[priority_group] += 1
    print(f"cluster1 candidates: {counts['cluster1_seed']}")
    print(f"cluster0 candidates: {counts['cluster0_mix']}")
    print(f"cluster2 candidates: {counts['cluster2_mix']}")


if __name__ == "__main__":
    main()
