import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    normalize_signals,
)
from features import extract_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clustering results for all pkls in a directory")
    parser.add_argument("--pkl_dir", required=True, help="Directory containing grouped pkl files")
    parser.add_argument("--clusters_csv", required=True, help="clusters_raw.csv or clusters_norm.csv (all data)")
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    parser.add_argument("--length_mode", default="crop", choices=["crop", "pad"], help="Length handling mode")
    parser.add_argument("--target_len", type=int, default=None, help="Optional target length")
    parser.add_argument("--mode", default="raw", choices=["raw", "norm"], help="Feature mode to visualize")
    parser.add_argument("--pca_components", type=int, default=2, help="PCA components for visualization")
    parser.add_argument("--max_points", type=int, default=8000, help="Max points to plot (subsample if larger)")
    return parser.parse_args()


def _read_clusters(path: Path) -> Dict[int, int]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["sample_id"])] = int(row["cluster_id"])
    return mapping


def _subsample(x: np.ndarray, labels: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_points:
        return x, labels
    idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
    return x[idx], labels[idx]


def main() -> None:
    args = parse_args()
    pkl_dir = Path(args.pkl_dir)
    if not pkl_dir.is_dir():
        raise FileNotFoundError(f"pkl_dir is not a directory: {pkl_dir}")

    records = load_records_from_dir(str(pkl_dir))
    signals, _, _ = build_signal_matrix(
        records,
        sample_length=args.target_len,
        length_mode=args.length_mode,
    )

    if args.mode == "norm":
        signals, _ = normalize_signals(signals, mode="per_sample")

    feats = extract_features(signals)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = StandardScaler().fit_transform(feats)
    if feats.shape[0] < 2 or feats.shape[1] < 2:
        emb = feats[:, :2] if feats.shape[1] >= 2 else np.c_[np.arange(feats.shape[0]), np.zeros(feats.shape[0])]
    else:
        pca = PCA(n_components=args.pca_components, random_state=42)
        emb = pca.fit_transform(feats)

    label_map = _read_clusters(Path(args.clusters_csv))
    labels = np.array([label_map.get(i, -999) for i in range(emb.shape[0])], dtype=int)

    emb, labels = _subsample(emb, labels, args.max_points)

    unique = sorted(set(labels))
    plt.figure(figsize=(9, 7))
    for lab in unique:
        mask = labels == lab
        if lab == -1:
            color = "lightgray"
        elif lab == -999:
            color = "black"
        else:
            color = None
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.7,
            c=color,
            label=f"cluster {lab}",
        )

    plt.title(f"PCA scatter ({args.mode})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
