"""
Plot cluster samples with raw, bandpass-filtered raw, and AGC signals side by side.

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
from dasQt.process.filter import bandpass  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cluster samples: raw vs bandpass vs AGC")
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
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate for bandpass")
    parser.add_argument("--freqmin", type=float, default=0.1, help="Bandpass low cutoff")
    parser.add_argument("--freqmax", type=float, default=2.0, help="Bandpass high cutoff")
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


def _meta_key(meta: Dict, with_source: bool = True) -> Tuple:
    key = (meta.get("veh_id"), meta.get("station"), meta.get("time"))
    if with_source:
        key = key + (meta.get("source"),)
    return key


def _build_index(records: List[Dict], target_len: Optional[int], length_mode: str) -> Dict:
    """
    Build index structures while keeping duplicate keys.

    Returns:
        {
          "signals": np.ndarray [N, L],
          "meta": List[Dict],
          "key_to_indices": Dict[key, List[int]],
          "occ_by_index": List[int],  # occurrence rank within same key
        }
    """
    signals, meta, _ = build_signal_matrix(records, sample_length=target_len, length_mode=length_mode)
    key_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    occ_by_index: List[int] = [0] * len(meta)
    for idx, m in enumerate(meta):
        key = _meta_key(m, with_source=True)
        occ_by_index[idx] = len(key_to_indices[key])
        key_to_indices[key].append(idx)
    return {
        "signals": signals,
        "meta": meta,
        "key_to_indices": dict(key_to_indices),
        "occ_by_index": occ_by_index,
    }


def _read_clusters_csv(csv_path: Path) -> Dict[int, List[Dict]]:
    """Return dict cluster_id -> list of row dicts."""
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            clusters[cid].append(row)
    return clusters


def _normalize_for_plot(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    return sig / peak if peak > 0 else sig


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_cluster(
    cluster_id: int,
    rows: List[Dict],
    raw_index: Dict,
    agc_index: Dict,
    out_dir: Path,
    sample_per_cluster: int,
    fs: float,
    freqmin: float,
    freqmax: float,
) -> None:
    # Build matched pairs
    raw_seen_counts: Dict[Tuple, int] = defaultdict(int)
    pairs: List[Tuple[np.ndarray, Optional[np.ndarray], str]] = []
    stat_total = 0
    stat_no_raw = 0
    stat_no_agc = 0
    stat_sampleid_hit = 0
    for row in rows:
        stat_total += 1
        key_src = (
            _cast(row.get("veh_id")),
            _cast(row.get("station")),
            _cast(row.get("time")),
            row.get("source"),
        )
        key_nosrc = key_src[:3]

        # 1) Raw: prefer exact sample_id to avoid ambiguity.
        raw_idx = None
        sample_id = _cast(row.get("sample_id"))
        if isinstance(sample_id, int) and 0 <= sample_id < len(raw_index["signals"]):
            raw_idx = sample_id
            stat_sampleid_hit += 1
        else:
            cand_raw = raw_index["key_to_indices"].get(key_src)
            if not cand_raw:
                # fallback for old CSV without source column or mismatched source format
                cand_raw = []
                for k, ids in raw_index["key_to_indices"].items():
                    if k[:3] == key_nosrc:
                        cand_raw.extend(ids)
            occ = raw_seen_counts[key_src]
            raw_idx = cand_raw[occ] if occ < len(cand_raw) else (cand_raw[-1] if cand_raw else None)
            raw_seen_counts[key_src] += 1

        raw_sig = raw_index["signals"][raw_idx] if raw_idx is not None else None
        if raw_sig is None:
            stat_no_raw += 1
            continue

        # 2) AGC: use the same key + occurrence order as raw.
        raw_meta = raw_index["meta"][raw_idx]
        raw_key = _meta_key(raw_meta, with_source=True)
        raw_occ = raw_index["occ_by_index"][raw_idx]
        cand_agc = agc_index["key_to_indices"].get(raw_key)
        if not cand_agc:
            cand_agc = []
            for k, ids in agc_index["key_to_indices"].items():
                if k[:3] == raw_key[:3]:
                    cand_agc.extend(ids)
        agc_idx = cand_agc[raw_occ] if raw_occ < len(cand_agc) else (cand_agc[-1] if cand_agc else None)
        agc_sig = agc_index["signals"][agc_idx] if agc_idx is not None else None
        if agc_sig is None:
            stat_no_agc += 1

        label = f"veh={row['veh_id']} sta={row['sta_name']} t={row['time']}"
        pairs.append((raw_sig, agc_sig, label))
        if len(pairs) >= sample_per_cluster:
            break

    if not pairs:
        print(f"  cluster {cluster_id}: no matching signals found")
        return

    n = len(pairs)
    fig, axes = plt.subplots(n, 3, figsize=(24, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Cluster {cluster_id}  (n={len(rows)})", y=0.999, fontsize=11)

    # Column headers
    axes[0][0].set_title("Raw", fontsize=10)
    axes[0][1].set_title(f"Bandpass ({freqmin}-{freqmax} Hz)", fontsize=10)
    axes[0][2].set_title("AGC", fontsize=10)

    for i, (raw_sig, agc_sig, label) in enumerate(pairs):
        ax_raw, ax_bp, ax_agc = axes[i][0], axes[i][1], axes[i][2]

        raw_plot = _normalize_for_plot(raw_sig)
        bp_sig = bandpass(
            raw_plot,
            fs,
            freqmin=freqmin,
            freqmax=freqmax,
            corners=4,
            zerophase=True,
            detrend=False,
            taper=False,
        )
        bp_plot = _normalize_for_plot(bp_sig)

        ax_raw.plot(raw_plot, lw=0.7, color="steelblue")
        ax_raw.set_ylabel(label, fontsize=6, rotation=0, labelpad=80, va="center")
        ax_raw.grid(alpha=0.3)
        ax_raw.tick_params(labelsize=7)

        ax_bp.plot(bp_plot, lw=0.7, color="seagreen")
        ax_bp.grid(alpha=0.3)
        ax_bp.tick_params(labelsize=7)

        if agc_sig is not None:
            ax_agc.plot(_normalize_for_plot(agc_sig), lw=0.7, color="darkorange")
        else:
            ax_agc.text(0.5, 0.5, "no AGC data", ha="center", va="center", transform=ax_agc.transAxes, fontsize=8)
        ax_agc.grid(alpha=0.3)
        ax_agc.tick_params(labelsize=7)

    axes[-1][0].set_xlabel("Sample Index")
    axes[-1][1].set_xlabel("Sample Index")
    axes[-1][2].set_xlabel("Sample Index")

    fig.tight_layout()
    out_path = out_dir / f"cluster_{cluster_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  saved: {out_path} "
        f"(rows={stat_total}, sample_id_hit={stat_sampleid_hit}, no_raw={stat_no_raw}, no_agc={stat_no_agc})"
    )


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
    raw_index = _build_index(raw_records, args.target_len, args.length_mode)
    print(f"  {len(raw_index['signals'])} raw signals loaded")

    print("Loading AGC signals...")
    agc_records = _load_records(Path(args.agc_pkl))
    agc_index = _build_index(agc_records, args.target_len, args.length_mode)
    print(f"  {len(agc_index['signals'])} AGC signals loaded")

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
            raw_index=raw_index,
            agc_index=agc_index,
            out_dir=out_dir,
            sample_per_cluster=args.sample_per_cluster,
            fs=args.fs,
            freqmin=args.freqmin,
            freqmax=args.freqmax,
        )

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
