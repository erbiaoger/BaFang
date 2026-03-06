import argparse
import sys
from pathlib import Path


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from cluster.agc_dataset import build_paired_dataset, save_paired_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired origin/AGC training set")
    parser.add_argument("--origin_pkl_dir", required=True, help="Origin/raw pkl directory or single pkl file")
    parser.add_argument("--agc_pkl_dir", required=True, help="AGC pkl directory or single pkl file")
    parser.add_argument("--clusters_csv", default=None, help="Optional old clustering CSV for cluster_id_old")
    parser.add_argument("--out_npz", required=True, help="Output manifest .npz path")
    parser.add_argument("--out_meta_csv", required=True, help="Output metadata CSV path")
    parser.add_argument("--target_len", type=int, default=4000, help="Sample length for crop/pad")
    parser.add_argument("--length_mode", default="crop", choices=["crop", "pad"], help="Length handling mode")
    parser.add_argument("--min_match_rate", type=float, default=0.98, help="Required AGC match rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals_origin, signals_agc, meta_rows, stats = build_paired_dataset(
        origin_path=Path(args.origin_pkl_dir),
        agc_path=Path(args.agc_pkl_dir),
        sample_length=args.target_len,
        length_mode=args.length_mode,
        clusters_csv=Path(args.clusters_csv) if args.clusters_csv else None,
    )

    save_paired_dataset(
        dataset_npz=Path(args.out_npz),
        meta_csv=Path(args.out_meta_csv),
        signals_origin=signals_origin,
        signals_agc=signals_agc,
        meta_rows=meta_rows,
        stats=stats,
    )

    print(f"saved manifest: {args.out_npz}")
    print(f"saved meta csv: {args.out_meta_csv}")
    print(
        f"AGC match rate: {stats['match_rate']:.4f} "
        f"({stats['num_matched']}/{stats['num_samples']})"
    )

    worst_sources = sorted(
        stats["per_source"].items(),
        key=lambda item: item[1]["matched"] / max(1, item[1]["total"]),
    )[:10]
    for source, source_stats in worst_sources:
        ratio = source_stats["matched"] / max(1, source_stats["total"])
        print(
            f"  source={source} matched={source_stats['matched']}/{source_stats['total']} "
            f"rate={ratio:.4f}"
        )

    if stats["match_rate"] < args.min_match_rate:
        raise RuntimeError(
            f"AGC match rate {stats['match_rate']:.4f} is below required {args.min_match_rate:.4f}"
        )


if __name__ == "__main__":
    main()
