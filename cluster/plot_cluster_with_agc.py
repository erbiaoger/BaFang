"""
Plot cluster samples with both raw and AGC signals side by side.

Usage:
    python cluster/plot_cluster_with_agc.py \
        --clusters_csv <out_dir>/clusters_raw.csv \
        --raw_pkl <path/to/raw.pkl or dir> \
        --agc_pkl <path/to/agc.pkl or dir> \
        --out_dir <output_dir> \
        [--cluster_ids 0 1 2] \
        [--sample_per_cluster 15] \
        [--target_len 4000]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# repo root path setup
# ---------------------------------------------------------------------------
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
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cluster samples: raw vs AGC side by side")
    parser.add_argument("--clusters_csv", required=True, help="Path to clusters_raw.csv or clusters_norm.csv")
    parser.add_argument("--raw_pkl", required=True, help="Raw pkl file or directory")
    parser.add_argument("--agc_pkl", required=True, help="AGC pkl file or directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument(
        "--cluster_ids", type=int, nargs="+", default=None,
        help="Which cluster IDs to plot (default: all found in CSV)",
    )
    parser.add_argument("--sample_per_cluster", type=int, default=15, help="Samples to show per cluster")
    parser.add_argument("--target_len", type=int, default=None, help="Crop/pad signals to this length")
    parser.add_argument("--length_mode", default="crop", choices=["crop", "pad"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_records(path: Path) -> List[Dict]:
    if path.is_dir():
        return load_records_from_dir(str(path))
    if path.is_file() and path.suffix.lower() == ".pkl":
        grouped = load_grouped_pkl(path)
        return flatten_grouped_records(grouped, path)
    raise FileNotFoundError(f"not found or unsupported: {path}")


def _build_lookup(records: List[Dict], target_len: Optional[int], length_mode: str) -> Dict[Tuple, np.ndarray]:
    """Build a dict keyed by (veh_id, station, time) -> signal array."""
    signals, meta, _ = build_signal_matrix(records, sample_length=target_len, length_mode=length_mode)
    lookup = {}
    for sig, m in zip(signals, meta):
        key = (m.get("veh_id"), m.get("station"), m.get("time"))
        lookup[key] = sig
    return lookup


def _read_clusters_csv(csv_path: Path) -> Dict[int, List[Dict]]:
    """Return dict cluster_id -> list of row dicts."""
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            clusters[cid].append(row)
    return clusters


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_cluster(
    cluster_id: int,
    rows: List[Dict],
    raw_lookup: Dict[Tuple, np.ndarray],
    agc_lookup: Dict[Tuple, np.ndarray],
    out_dir: Path,
    sample_per_cluster: int,
) -> None:
    # Build matched pairs
    pairs: List[Tuple[np.ndarray, Optional[np.ndarray], str]] = []
    for row in rows:
        key = (_cast(row["veh_id"]), _cast(row["station"]), _cast(row["time"]))
        raw_sig = raw_lookup.get(key)
        agc_sig = agc_lookup.get(key)
        if raw_sig is None:
            continue
        label = f"veh={row['veh_id']} sta={row['sta_name']} t={row['time']}"
        pairs.append((raw_sig, agc_sig, label))
        if len(pairs) >= sample_per_cluster:
            break

    if not pairs:
        print(f"  cluster {cluster_id}: no matching signals found")
        return

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(16, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Cluster {cluster_id}  (n={len(rows)})", y=0.999, fontsize=11)

    # Column headers
    axes[0][0].set_title("Raw", fontsize=10)
    axes[0][1].set_title("AGC", fontsize=10)

    for i, (raw_sig, agc_sig, label) in enumerate(pairs):
        ax_raw, ax_agc = axes[i][0], axes[i][1]

        ax_raw.plot(raw_sig, lw=0.7, color="steelblue")
        ax_raw.set_ylabel(label, fontsize=6, rotation=0, labelpad=80, va="center")
        ax_raw.grid(alpha=0.3)
        ax_raw.tick_params(labelsize=7)

        if agc_sig is not None:
            ax_agc.plot(agc_sig, lw=0.7, color="darkorange")
        else:
            ax_agc.text(0.5, 0.5, "no AGC data", ha="center", va="center", transform=ax_agc.transAxes, fontsize=8)
        ax_agc.grid(alpha=0.3)
        ax_agc.tick_params(labelsize=7)

    axes[-1][0].set_xlabel("Sample Index")
    axes[-1][1].set_xlabel("Sample Index")

    fig.tight_layout()
    out_path = out_dir / f"cluster_{cluster_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def _cast(val: str):
    """Try to cast CSV string value to int, else keep as string."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw signals...")
    raw_records = _load_records(Path(args.raw_pkl))
    raw_lookup = _build_lookup(raw_records, args.target_len, args.length_mode)
    print(f"  {len(raw_lookup)} raw signals loaded")

    print("Loading AGC signals...")
    agc_records = _load_records(Path(args.agc_pkl))
    agc_lookup = _build_lookup(agc_records, args.target_len, args.length_mode)
    print(f"  {len(agc_lookup)} AGC signals loaded")

    print("Reading clusters CSV...")
    clusters = _read_clusters_csv(Path(args.clusters_csv))

    cluster_ids = args.cluster_ids if args.cluster_ids is not None else sorted(clusters.keys())
    print(f"Plotting clusters: {cluster_ids}")

    for cid in cluster_ids:
        rows = clusters.get(cid, [])
        print(f"  cluster {cid}: {len(rows)} samples")
        _plot_cluster(
            cluster_id=cid,
            rows=rows,
            raw_lookup=raw_lookup,
            agc_lookup=agc_lookup,
            out_dir=out_dir,
            sample_per_cluster=args.sample_per_cluster,
        )

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
