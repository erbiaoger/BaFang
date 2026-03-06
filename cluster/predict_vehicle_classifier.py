import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from cluster.agc_dataset import load_paired_dataset  # noqa: E402
from cluster.agc_features import apply_smallcar_rule, extract_agc_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict large/small vehicle labels from paired AGC dataset")
    parser.add_argument("--dataset_npz", required=True, help="Dataset manifest produced by build_agc_training_set.py")
    parser.add_argument("--meta_csv", required=True, help="Metadata CSV produced by build_agc_training_set.py")
    parser.add_argument("--model_dir", required=True, help="Directory containing model.joblib")
    parser.add_argument("--out_csv", required=True, help="Output prediction CSV")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--apply_smallcar_rule_fallback", action="store_true")
    return parser.parse_args()


def _prob_small(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    small_idx = classes.index(1)
    return probs[:, small_idx]


def _confidence_bucket(prob_small: np.ndarray, small_threshold: float, large_threshold: float) -> np.ndarray:
    buckets = np.full(prob_small.shape, "uncertain", dtype=object)
    buckets[prob_small >= small_threshold] = "small_high_conf"
    buckets[prob_small <= large_threshold] = "large_high_conf"
    return buckets


def _pred_label(prob_small: np.ndarray, small_threshold: float, large_threshold: float) -> np.ndarray:
    labels = np.full(prob_small.shape, "uncertain", dtype=object)
    labels[prob_small >= small_threshold] = "small"
    labels[prob_small <= large_threshold] = "large"
    return labels


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    with open(model_dir / "model.joblib", "rb") as f:
        payload = pickle.load(f)

    signals_origin, signals_agc, meta_rows, dataset_info = load_paired_dataset(
        Path(args.dataset_npz),
        Path(args.meta_csv),
        mmap_mode="r",
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    review_dir = out_csv.parent / "review_samples"
    for name in ["small_high_conf", "large_high_conf", "uncertain"]:
        (review_dir / name).mkdir(parents=True, exist_ok=True)

    summary = Counter()
    fieldnames = [
        "sample_id",
        "pred_label",
        "prob_small",
        "confidence_bucket",
        "rule_smallcar_high_conf",
        "veh_id",
        "station",
        "sta_name",
        "time",
        "source_origin",
        "source_agc",
        "cluster_id_old",
        "agc_matched",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        n = len(meta_rows)
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            batch_origin = signals_origin[start:end]
            batch_agc = signals_agc[start:end]
            x_batch = extract_agc_features(batch_agc, batch_origin, templates=payload["templates"])
            prob_small = _prob_small(payload["model"], x_batch)
            pred_label = _pred_label(prob_small, payload["small_threshold"], payload["large_threshold"])
            confidence_bucket = _confidence_bucket(prob_small, payload["small_threshold"], payload["large_threshold"])
            rule_mask = apply_smallcar_rule(x_batch, payload["rule_config"])

            if args.apply_smallcar_rule_fallback:
                fallback_mask = (pred_label == "uncertain") & rule_mask & (prob_small >= 0.55)
                pred_label[fallback_mask] = "small"
                confidence_bucket[fallback_mask] = "small_high_conf"

            for row, prob, label, bucket, rule_flag in zip(
                meta_rows[start:end],
                prob_small,
                pred_label,
                confidence_bucket,
                rule_mask,
            ):
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "pred_label": label,
                        "prob_small": f"{float(prob):.6f}",
                        "confidence_bucket": bucket,
                        "rule_smallcar_high_conf": int(bool(rule_flag)),
                        "veh_id": row.get("veh_id", ""),
                        "station": row.get("station", ""),
                        "sta_name": row.get("sta_name", ""),
                        "time": row.get("time", ""),
                        "source_origin": row.get("source_origin", ""),
                        "source_agc": row.get("source_agc", ""),
                        "cluster_id_old": row.get("cluster_id_old", ""),
                        "agc_matched": row.get("agc_matched", ""),
                    }
                )
                summary[f"pred_label:{label}"] += 1
                summary[f"bucket:{bucket}"] += 1
                summary[f"matched:{row.get('agc_matched', 0)}"] += 1

    summary_path = out_csv.with_name(f"{out_csv.stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_info, "counts": dict(summary)}, f, ensure_ascii=False, indent=2)

    print(f"saved predictions: {out_csv}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
