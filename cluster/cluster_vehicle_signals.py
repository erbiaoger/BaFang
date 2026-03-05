import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Limit BLAS threads to avoid OpenBLAS thread/memory explosion
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from diffusion.diffusion_dataset import (  # noqa: E402
    build_signal_matrix,
    flatten_grouped_records,
    load_grouped_pkl,
    load_records_from_dir,
    normalize_signals,
)
from features import extract_features, feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster vehicle signals into size classes")
    parser.add_argument("--pkl_dir", required=True, help="Directory with grouped pkl files OR a single .pkl file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--length_mode", default="crop", choices=["crop", "pad"], help="Length handling mode")
    parser.add_argument("--target_len", type=int, default=None, help="Optional target length")
    parser.add_argument("--use_pca", action="store_true", help="Enable PCA dimensionality reduction")
    parser.add_argument("--algo", default="hdbscan", choices=["hdbscan", "kmeans"], help="Clustering algorithm")
    parser.add_argument("--num_classes", type=int, default=3, help="Final number of classes (default: 3)")
    parser.add_argument(
        "--per_file",
        action="store_true",
        help="If set and pkl_dir is a directory, cluster each .pkl separately into subfolders",
    )
    parser.add_argument("--min_cluster_size", type=int, default=30, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument("--k_range", default="3,6", help="KMeans range as 'min,max'")
    parser.add_argument("--sample_per_cluster", type=int, default=15, help="Samples per cluster for visualization")
    parser.add_argument(
        "--preassign_other",
        action="store_true",
        help="Pre-assign tonal/very-low-energy signals to class 2 (other) before clustering",
    )
    parser.add_argument("--rms_floor", type=float, default=1e-8, help="RMS below this -> other")
    parser.add_argument("--flatness_max", type=float, default=0.02, help="Spectral flatness below this -> tonal")
    parser.add_argument("--bandwidth_max", type=float, default=0.02, help="Spectral bandwidth below this -> tonal")
    return parser.parse_args()


def _load_records(path: Path) -> List[Dict]:
    if path.is_dir():
        return load_records_from_dir(str(path))
    if path.is_file() and path.suffix.lower() == ".pkl":
        grouped = load_grouped_pkl(path)
        return flatten_grouped_records(grouped, path)
    raise FileNotFoundError(f"pkl_dir not found or unsupported file: {path}")


def _list_pkl_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".pkl":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.pkl"))
    return []


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _pca_reduce(x: np.ndarray, enable: bool) -> Tuple[np.ndarray, Optional[Dict]]:
    n_samples = x.shape[0]
    n_features = x.shape[1] if x.ndim > 1 else 1
    if not enable or n_features < 5 or n_samples < 2:
        return x, None
    pca_tmp = PCA(n_components=0.95, svd_solver="full", random_state=42)
    pca_tmp.fit(x)
    k95 = pca_tmp.n_components_
    k = min(30, max(10, int(k95)))
    k = min(k, n_samples, n_features)
    if k >= n_features or k < 2:
        return x, None
    pca = PCA(n_components=k, svd_solver="full", random_state=42)
    x_pca = pca.fit_transform(x)
    info = {
        "n_components": k,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return x_pca, info


def _score_labels(x: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    unique = np.unique(labels[mask])
    if unique.size < 2:
        return -1.0
    try:
        return float(silhouette_score(x[mask], labels[mask]))
    except Exception:
        return -1.0


def _hdbscan_grid(
    x: np.ndarray,
    min_cluster_sizes: List[int],
    min_samples_list: List[int],
) -> Tuple[np.ndarray, Dict]:
    if hdbscan is None:
        raise RuntimeError("hdbscan is not installed; install it or use --algo kmeans")

    best = None
    best_score = -1e9
    best_labels = None

    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(x)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            score = _score_labels(x, labels)
            # prefer 3-4 clusters
            penalty = abs(n_clusters - 3.5) * 0.1
            total = score - penalty
            if total > best_score:
                best_score = total
                best = {"min_cluster_size": mcs, "min_samples": ms, "silhouette": score, "n_clusters": n_clusters}
                best_labels = labels

    return best_labels, best or {}


def _kmeans_best(x: np.ndarray, k_min: int, k_max: int) -> Tuple[np.ndarray, Dict]:
    best_score = -1e9
    best = None
    best_labels = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(x)
        score = _score_labels(x, labels)
        if score > best_score:
            best_score = score
            best = {"k": k, "silhouette": score}
            best_labels = labels
    return best_labels, best or {}


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_cluster_samples(
    signals: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    sample_per_cluster: int,
    title_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(labels))
    for lab in unique:
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        n = min(sample_per_cluster, idx.size)
        pick = idx[:n]
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.0 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, si in enumerate(pick):
            axes[i].plot(signals[si], lw=0.8)
            axes[i].set_ylabel(f"#{si}")
            axes[i].grid(alpha=0.3)
        axes[-1].set_xlabel("Sample Index")
        fig.suptitle(f"{title_prefix} cluster {lab}", y=0.995)
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{lab}.png", dpi=160)
        plt.close(fig)


def _summarize(labels: np.ndarray) -> Dict:
    unique = sorted(set(labels))
    counts = {str(lab): int(np.sum(labels == lab)) for lab in unique}
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    noise_ratio = float(np.mean(labels == -1)) if labels.size else 0.0
    return {"counts": counts, "n_clusters": n_clusters, "noise_ratio": noise_ratio}


def _get_feat_idx() -> Dict[str, int]:
    return {name: i for i, name in enumerate(feature_names())}


def _preassign_other(feats: np.ndarray, rms_floor: float, flatness_max: float, bandwidth_max: float) -> np.ndarray:
    idx = _get_feat_idx()
    rms = feats[:, idx["rms"]]
    flatness = feats[:, idx["spectral_flatness"]]
    bandwidth = feats[:, idx["spectral_bandwidth"]]
    too_low = rms <= rms_floor
    tonal = (flatness <= flatness_max) & (bandwidth <= bandwidth_max)
    return too_low | tonal


def _map_to_three_classes(labels: np.ndarray, signals_for_size: np.ndarray) -> Tuple[np.ndarray, Dict]:
    # Use top-2 largest clusters as big/small, all others + noise -> other (2).
    unique = [lab for lab in sorted(set(labels)) if lab != -1]
    if len(unique) < 2:
        mapped = np.full(labels.shape, 2, dtype=int)
        return mapped, {"strategy": "fallback_all_other"}

    sizes = {lab: int(np.sum(labels == lab)) for lab in unique}
    top2 = sorted(unique, key=lambda l: sizes[l], reverse=True)[:2]
    # Determine big/small by mean RMS on raw signals
    def _mean_rms(lab: int) -> float:
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            return 0.0
        x = signals_for_size[idx]
        rms = np.sqrt(np.mean(np.square(x), axis=1))
        return float(np.mean(rms))

    rms0 = _mean_rms(top2[0])
    rms1 = _mean_rms(top2[1])
    if rms0 >= rms1:
        big_lab, small_lab = top2[0], top2[1]
    else:
        big_lab, small_lab = top2[1], top2[0]

    mapped = np.full(labels.shape, 2, dtype=int)  # other
    mapped[labels == big_lab] = 0
    mapped[labels == small_lab] = 1
    return mapped, {
        "strategy": "top2_by_size_then_rms",
        "big_cluster": int(big_lab),
        "small_cluster": int(small_lab),
    }


def _run_mode(
    mode_name: str,
    signals: np.ndarray,
    signals_for_size: np.ndarray,
    meta: List[Dict],
    out_dir: Path,
    algo: str,
    use_pca: bool,
    num_classes: int,
    min_cluster_size: int,
    min_samples: int,
    k_range: Tuple[int, int],
    sample_per_cluster: int,
    preassign_other: bool,
    rms_floor: float,
    flatness_max: float,
    bandwidth_max: float,
) -> None:
    feats = extract_features(signals)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    pre_mask = None
    if preassign_other:
        pre_mask = _preassign_other(feats, rms_floor=rms_floor, flatness_max=flatness_max, bandwidth_max=bandwidth_max)
    if pre_mask is not None and np.any(pre_mask):
        keep = ~pre_mask
        feats_use = feats[keep]
        signals_use = signals[keep]
        meta_use = [m for i, m in enumerate(meta) if keep[i]]
        keep_index = np.where(keep)[0]
    else:
        feats_use = feats
        signals_use = signals
        meta_use = meta
        keep_index = None

    if feats_use.shape[0] == 0:
        labels = np.full(signals.shape[0], 2, dtype=int)
        summary = _summarize(labels)
        summary.update(
            {
                "mode": mode_name,
                "algo": "none",
                "pca": None,
                "feature_names": feature_names(),
                "best_params": {"note": "all_preassigned_or_empty"},
                "mapping": {"strategy": "all_other"},
                "preassign_other": {
                    "enabled": bool(preassign_other),
                    "rms_floor": rms_floor,
                    "flatness_max": flatness_max,
                    "bandwidth_max": bandwidth_max,
                    "num_preassigned": int(np.sum(pre_mask)) if pre_mask is not None else 0,
                },
            }
        )
        rows = []
        for i, m in enumerate(meta):
            rows.append(
                {
                    "sample_id": i,
                    "cluster_id": int(labels[i]),
                    "veh_id": m.get("veh_id"),
                    "station": m.get("station"),
                    "sta_name": m.get("sta_name"),
                    "time": m.get("time"),
                    "source": m.get("source"),
                }
            )
        csv_path = out_dir / f"clusters_{mode_name}.csv"
        summary_path = out_dir / f"cluster_summary_{mode_name}.json"
        samples_dir = out_dir / "cluster_samples" / mode_name
        _write_csv(csv_path, rows)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _plot_cluster_samples(signals, labels, samples_dir, sample_per_cluster, title_prefix=mode_name)
        return

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats_use)
    feats_pca, pca_info = _pca_reduce(feats_scaled, use_pca)

    n_samples = feats_pca.shape[0]
    if n_samples < 3:
        labels = np.full(n_samples, 2, dtype=int)
        best = {"note": "too_few_samples"}
        algo_used = "none"
    else:
        algo_used = algo
        if algo == "hdbscan":
            if n_samples < min_cluster_size or n_samples <= min_samples:
                labels, best = _kmeans_best(feats_pca, 2, min(3, n_samples))
                algo_used = "kmeans_fallback_small"
            else:
                labels, best = _hdbscan_grid(
                    feats_pca,
                    min_cluster_sizes=[min_cluster_size, max(5, min_cluster_size // 2), min_cluster_size * 2],
                    min_samples_list=[min_samples, max(3, min_samples // 2)],
                )
                if labels is None:
                    raise RuntimeError("HDBSCAN failed to produce labels")
                if best and (best.get("n_clusters", 0) < 2 or best.get("n_clusters", 0) > 6):
                    labels, best = _kmeans_best(feats_pca, k_range[0], k_range[1])
                    algo_used = "kmeans_fallback"
        else:
            if num_classes == 3:
                labels, best = _kmeans_best(feats_pca, 3, 3)
            else:
                labels, best = _kmeans_best(feats_pca, k_range[0], k_range[1])
            algo_used = "kmeans"

    map_info = None
    if num_classes == 3:
        labels, map_info = _map_to_three_classes(labels, signals_for_size[keep_index] if keep_index is not None else signals_for_size)

    if keep_index is not None:
        full_labels = np.full(signals.shape[0], 2, dtype=int)
        full_labels[keep_index] = labels
        labels = full_labels

    summary = _summarize(labels)
    summary.update(
        {
            "mode": mode_name,
            "algo": algo_used,
            "pca": pca_info,
            "feature_names": feature_names(),
            "best_params": best,
            "mapping": map_info,
            "preassign_other": {
                "enabled": bool(preassign_other),
                "rms_floor": rms_floor,
                "flatness_max": flatness_max,
                "bandwidth_max": bandwidth_max,
                "num_preassigned": int(np.sum(pre_mask)) if pre_mask is not None else 0,
            },
        }
    )

    rows = []
    for i, m in enumerate(meta):
        rows.append(
            {
                "sample_id": i,
                "cluster_id": int(labels[i]),
                "veh_id": m.get("veh_id"),
                "station": m.get("station"),
                "sta_name": m.get("sta_name"),
                "time": m.get("time"),
                "source": m.get("source"),
            }
        )

    csv_path = out_dir / f"clusters_{mode_name}.csv"
    summary_path = out_dir / f"cluster_summary_{mode_name}.json"
    samples_dir = out_dir / "cluster_samples" / mode_name

    _write_csv(csv_path, rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _plot_cluster_samples(signals, labels, samples_dir, sample_per_cluster, title_prefix=mode_name)


def _run_for_records(
    records: List[Dict],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    _ensure_out_dir(out_dir)
    signals, meta, info = build_signal_matrix(
        records,
        sample_length=args.target_len,
        length_mode=args.length_mode,
    )

    k_min, k_max = [int(x) for x in args.k_range.split(",")]

    _run_mode(
        mode_name="raw",
        signals=signals,
        signals_for_size=signals,
        meta=meta,
        out_dir=out_dir,
        algo=args.algo,
        use_pca=args.use_pca,
        num_classes=args.num_classes,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        k_range=(k_min, k_max),
        sample_per_cluster=args.sample_per_cluster,
        preassign_other=args.preassign_other,
        rms_floor=args.rms_floor,
        flatness_max=args.flatness_max,
        bandwidth_max=args.bandwidth_max,
    )

    signals_norm, _ = normalize_signals(signals, mode="per_sample")
    _run_mode(
        mode_name="norm",
        signals=signals_norm,
        signals_for_size=signals,
        meta=meta,
        out_dir=out_dir,
        algo=args.algo,
        use_pca=args.use_pca,
        num_classes=args.num_classes,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        k_range=(k_min, k_max),
        sample_per_cluster=args.sample_per_cluster,
        preassign_other=args.preassign_other,
        rms_floor=args.rms_floor,
        flatness_max=args.flatness_max,
        bandwidth_max=args.bandwidth_max,
    )

    info_path = out_dir / "data_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"saved outputs to: {out_dir}")


def main() -> None:
    args = parse_args()
    pkl_path = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)

    if args.per_file and pkl_path.is_dir():
        pkl_files = _list_pkl_files(pkl_path)
        if not pkl_files:
            raise FileNotFoundError(f"no .pkl files found under: {pkl_path}")
        for pkl in pkl_files:
            records = _load_records(pkl)
            sub_out = out_dir / pkl.stem
            _run_for_records(records, sub_out, args)
        return

    records = _load_records(pkl_path)
    _run_for_records(records, out_dir, args)


if __name__ == "__main__":
    main()
