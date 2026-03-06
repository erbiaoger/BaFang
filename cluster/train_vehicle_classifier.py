import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from cluster.agc_dataset import load_paired_dataset  # noqa: E402
from cluster.agc_features import (  # noqa: E402
    agc_feature_names,
    apply_smallcar_rule,
    build_smallcar_templates,
    derive_smallcar_rule_config,
    extract_agc_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AGC-driven vehicle classifier")
    parser.add_argument("--dataset_npz", required=True, help="Dataset manifest produced by build_agc_training_set.py")
    parser.add_argument("--meta_csv", required=True, help="Metadata CSV produced by build_agc_training_set.py")
    parser.add_argument("--labels_csv", required=True, help="Manual labels CSV with sample_id,label")
    parser.add_argument("--out_dir", required=True, help="Output model directory")
    parser.add_argument("--model", default="auto", choices=["auto", "rf", "xgb"], help="Classifier backend")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--small_threshold", type=float, default=0.80)
    parser.add_argument("--large_threshold", type=float, default=0.20)
    parser.add_argument("--num_templates", type=int, default=1)
    parser.add_argument("--max_template_samples", type=int, default=30)
    return parser.parse_args()


def _read_labels_csv(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id")
            label = (row.get("label") or "").strip().lower()
            if not sample_id or label not in {"small", "large"}:
                continue
            labels[int(sample_id)] = label
    return labels


def _prepare_labeled_rows(meta_rows: List[Dict], label_map: Dict[int, str]) -> List[Dict]:
    labeled_rows = []
    for row in meta_rows:
        sample_id = int(row["sample_id"])
        if sample_id not in label_map:
            continue
        if int(row.get("agc_matched", 0) or 0) != 1:
            continue
        item = dict(row)
        item["label"] = label_map[sample_id]
        labeled_rows.append(item)
    return labeled_rows


def _ensure_binary_coverage(y: np.ndarray) -> bool:
    return set(np.unique(y).tolist()) == {0, 1}


def _group_split_indices(labeled_rows: List[Dict], random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([str(row["veh_id"]) for row in labeled_rows])
    y = np.asarray([1 if row["label"] == "small" else 0 for row in labeled_rows], dtype=int)
    idx = np.arange(len(labeled_rows))

    for offset in range(100):
        gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state + offset)
        train_pos, temp_pos = next(gss_train.split(idx, y, groups))
        if temp_pos.size < 2 or not _ensure_binary_coverage(y[train_pos]):
            continue

        temp_groups = groups[temp_pos]
        temp_y = y[temp_pos]
        if np.unique(temp_groups).size < 2:
            continue

        gss_eval = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state + 100 + offset)
        val_rel, test_rel = next(gss_eval.split(temp_pos, temp_y, temp_groups))
        val_pos = temp_pos[val_rel]
        test_pos = temp_pos[test_rel]

        if not _ensure_binary_coverage(y[val_pos]) or not _ensure_binary_coverage(y[test_pos]):
            continue
        return train_pos, val_pos, test_pos

    raise RuntimeError("Unable to find a group split with both classes in train/val/test")


def _build_model(model_name: str, y_train: np.ndarray, random_state: int):
    if model_name in {"auto", "xgb"}:
        try:
            from xgboost import XGBClassifier

            pos = max(1, int(np.sum(y_train == 1)))
            neg = max(1, int(np.sum(y_train == 0)))
            return (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    scale_pos_weight=float(neg / pos),
                ),
            )
        except Exception:
            if model_name == "xgb":
                raise

    return (
        "rf",
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    )


def _prob_small(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.shape[1] != 2:
        raise RuntimeError("expected binary classifier probabilities")
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


def _evaluate_split(y_true: np.ndarray, prob_small: np.ndarray, small_threshold: float, large_threshold: float) -> Dict:
    pred_binary = (prob_small >= 0.50).astype(int)
    buckets = _confidence_bucket(prob_small, small_threshold, large_threshold)
    high_conf = buckets != "uncertain"
    pred_high_conf = np.where(prob_small >= small_threshold, 1, 0)

    metrics = {
        "num_samples": int(y_true.size),
        "small_precision": float(precision_score(y_true, pred_binary, pos_label=1, zero_division=0)),
        "small_recall": float(recall_score(y_true, pred_binary, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred_binary, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred_binary)),
        "average_precision": float(average_precision_score(y_true, prob_small)),
        "confusion_matrix_binary": confusion_matrix(y_true, pred_binary, labels=[0, 1]).tolist(),
        "high_conf_coverage": float(np.mean(high_conf)) if y_true.size else 0.0,
        "high_conf_accuracy": float(accuracy_score(y_true[high_conf], pred_high_conf[high_conf])) if np.any(high_conf) else 0.0,
        "high_conf_small_precision": float(
            precision_score(y_true[prob_small >= small_threshold], np.ones(np.sum(prob_small >= small_threshold), dtype=int), zero_division=0)
        )
        if np.any(prob_small >= small_threshold)
        else 0.0,
    }
    return metrics


def _plot_confusion_matrices(out_path: Path, val_cm: np.ndarray, test_cm: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, cm, title in zip(axes, [val_cm, test_cm], ["Validation", "Test"]):
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_title(title)
        ax.set_xticks([0, 1], labels=["large", "small"])
        ax.set_yticks([0, 1], labels=["large", "small"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_prediction_rows(
    path: Path,
    rows: List[Dict],
    prob_small: np.ndarray,
    pred_label: np.ndarray,
    confidence_bucket: np.ndarray,
    rule_mask: np.ndarray,
    split_name: str,
) -> None:
    fieldnames = [
        "sample_id",
        "split",
        "true_label",
        "prob_small",
        "pred_label",
        "confidence_bucket",
        "rule_smallcar_high_conf",
        "veh_id",
        "station",
        "sta_name",
        "time",
        "cluster_id_old",
        "source_origin",
        "source_agc",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, prob, pred, bucket, rule_flag in zip(rows, prob_small, pred_label, confidence_bucket, rule_mask):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "split": split_name,
                    "true_label": row["label"],
                    "prob_small": f"{float(prob):.6f}",
                    "pred_label": pred,
                    "confidence_bucket": bucket,
                    "rule_smallcar_high_conf": int(bool(rule_flag)),
                    "veh_id": row.get("veh_id", ""),
                    "station": row.get("station", ""),
                    "sta_name": row.get("sta_name", ""),
                    "time": row.get("time", ""),
                    "cluster_id_old": row.get("cluster_id_old", ""),
                    "source_origin": row.get("source_origin", ""),
                    "source_agc": row.get("source_agc", ""),
                }
            )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_origin, signals_agc, meta_rows, dataset_info = load_paired_dataset(
        Path(args.dataset_npz),
        Path(args.meta_csv),
        mmap_mode="r",
    )
    label_map = _read_labels_csv(Path(args.labels_csv))
    labeled_rows = _prepare_labeled_rows(meta_rows, label_map)
    if len(labeled_rows) < 20:
        raise RuntimeError("need at least 20 matched labeled samples to train")

    train_pos, val_pos, test_pos = _group_split_indices(labeled_rows, args.random_state)
    y_all = np.asarray([1 if row["label"] == "small" else 0 for row in labeled_rows], dtype=int)
    labeled_sample_ids = np.asarray([int(row["sample_id"]) for row in labeled_rows], dtype=int)

    train_sample_ids = labeled_sample_ids[train_pos]
    val_sample_ids = labeled_sample_ids[val_pos]
    test_sample_ids = labeled_sample_ids[test_pos]

    templates = build_smallcar_templates(
        signals_agc[train_sample_ids][y_all[train_pos] == 1],
        max_templates=args.num_templates,
        max_samples=args.max_template_samples,
    )

    x_train = extract_agc_features(signals_agc[train_sample_ids], signals_origin[train_sample_ids], templates=templates)
    x_val = extract_agc_features(signals_agc[val_sample_ids], signals_origin[val_sample_ids], templates=templates)
    x_test = extract_agc_features(signals_agc[test_sample_ids], signals_origin[test_sample_ids], templates=templates)

    model_kind, model = _build_model(args.model, y_all[train_pos], args.random_state)
    model.fit(x_train, y_all[train_pos])

    val_prob_small = _prob_small(model, x_val)
    test_prob_small = _prob_small(model, x_test)

    rule_config = derive_smallcar_rule_config(x_train, y_all[train_pos] == 1)
    val_rule_mask = apply_smallcar_rule(x_val, rule_config)
    test_rule_mask = apply_smallcar_rule(x_test, rule_config)

    val_pred_label = _pred_label(val_prob_small, args.small_threshold, args.large_threshold)
    test_pred_label = _pred_label(test_prob_small, args.small_threshold, args.large_threshold)
    val_confidence = _confidence_bucket(val_prob_small, args.small_threshold, args.large_threshold)
    test_confidence = _confidence_bucket(test_prob_small, args.small_threshold, args.large_threshold)

    val_metrics = _evaluate_split(y_all[val_pos], val_prob_small, args.small_threshold, args.large_threshold)
    test_metrics = _evaluate_split(y_all[test_pos], test_prob_small, args.small_threshold, args.large_threshold)

    feature_names = agc_feature_names()
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    else:
        importances = np.zeros(len(feature_names), dtype=float)

    feature_rows = sorted(
        [{"feature": name, "importance": float(score)} for name, score in zip(feature_names, importances)],
        key=lambda row: row["importance"],
        reverse=True,
    )
    with open(out_dir / "feature_importance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(feature_rows)

    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    metrics = {
        "dataset": dataset_info,
        "model_kind": model_kind,
        "num_labeled": len(labeled_rows),
        "num_train": int(train_pos.size),
        "num_val": int(val_pos.size),
        "num_test": int(test_pos.size),
        "small_threshold": args.small_threshold,
        "large_threshold": args.large_threshold,
        "rule_config": rule_config,
        "train_label_counts": {
            "large": int(np.sum(y_all[train_pos] == 0)),
            "small": int(np.sum(y_all[train_pos] == 1)),
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    _plot_confusion_matrices(
        out_dir / "confusion_matrix.png",
        np.asarray(val_metrics["confusion_matrix_binary"], dtype=int),
        np.asarray(test_metrics["confusion_matrix_binary"], dtype=int),
    )

    _save_prediction_rows(
        out_dir / "val_predictions.csv",
        [labeled_rows[i] for i in val_pos],
        val_prob_small,
        val_pred_label,
        val_confidence,
        val_rule_mask,
        split_name="val",
    )
    _save_prediction_rows(
        out_dir / "test_predictions.csv",
        [labeled_rows[i] for i in test_pos],
        test_prob_small,
        test_pred_label,
        test_confidence,
        test_rule_mask,
        split_name="test",
    )

    payload = {
        "model_kind": model_kind,
        "model": model,
        "feature_names": feature_names,
        "templates": templates.astype(np.float32),
        "small_threshold": float(args.small_threshold),
        "large_threshold": float(args.large_threshold),
        "rule_config": rule_config,
    }
    with open(out_dir / "model.joblib", "wb") as f:
        pickle.dump(payload, f)

    print(f"saved model artifacts to: {out_dir}")
    print(f"validation small precision: {val_metrics['small_precision']:.4f}")
    print(f"test small precision: {test_metrics['small_precision']:.4f}")


if __name__ == "__main__":
    main()
