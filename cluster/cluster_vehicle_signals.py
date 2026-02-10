import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from diffusion.diffusion_dataset import (
    build_signal_matrix,
    load_records_from_dir,
    normalize_signals,
)
from vehicle_signal.features import extract_features, feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster vehicle signals into size classes")
    parser.add_argument("--pkl_dir", required=True, help="Directory with grouped pkl files")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--length_mode", default="crop", choices=["crop", "pad"], help="Length handling mode")
    parser.add_argument("--target_len", type=int, default=None, help="Optional target length")
    parser.add_argument("--use_pca", action="store_true", help="Enable PCA dimensionality reduction")
    parser.add_argument("--algo", default="hdbscan", choices=["hdbscan", "kmeans"], help="Clustering algorithm")
    parser.add_argument("--min_cluster_size", type=int, default=30, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument("--k_range", default="3,6", help="KMeans range as 'min,max'")
    parser.add_argument("--sample_per_cluster", type=int, default=15, help="Samples per cluster for visualization")
    return parser.parse_args()


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _pca_reduce(x: np.ndarray, enable: bool) -> Tuple[np.ndarray, Optional[Dict]]:
    if not enable or x.shape[1] < 5:
        return x, None
    pca_tmp = PCA(n_components=0.95, svd_solver="full", random_state=42)
    pca_tmp.fit(x)
    k95 = pca_tmp.n_components_
    k = min(30, max(10, int(k95)))
    if k >= x.shape[1]:
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


def _run_mode(
    mode_name: str,
    signals: np.ndarray,
    meta: List[Dict],
    out_dir: Path,
    algo: str,
    use_pca: bool,
    min_cluster_size: int,
    min_samples: int,
    k_range: Tuple[int, int],
    sample_per_cluster: int,
) -> None:
    feats = extract_features(signals)
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    feats_pca, pca_info = _pca_reduce(feats_scaled, use_pca)

    algo_used = algo
    if algo == "hdbscan":
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
        labels, best = _kmeans_best(feats_pca, k_range[0], k_range[1])

    summary = _summarize(labels)
    summary.update(
        {
            "mode": mode_name,
            "algo": algo_used,
            "pca": pca_info,
            "feature_names": feature_names(),
            "best_params": best,
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    records = load_records_from_dir(args.pkl_dir)
    signals, meta, info = build_signal_matrix(
        records,
        sample_length=args.target_len,
        length_mode=args.length_mode,
    )

    k_min, k_max = [int(x) for x in args.k_range.split(",")]

    # Raw mode (no per-sample normalization)
    _run_mode(
        mode_name="raw",
        signals=signals,
        meta=meta,
        out_dir=out_dir,
        algo=args.algo,
        use_pca=args.use_pca,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        k_range=(k_min, k_max),
        sample_per_cluster=args.sample_per_cluster,
    )

    # Normalized mode (per-sample)
    signals_norm, _ = normalize_signals(signals, mode="per_sample")
    _run_mode(
        mode_name="norm",
        signals=signals_norm,
        meta=meta,
        out_dir=out_dir,
        algo=args.algo,
        use_pca=args.use_pca,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        k_range=(k_min, k_max),
        sample_per_cluster=args.sample_per_cluster,
    )

    info_path = out_dir / "data_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
