import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot generated signals from synth_signals.npz")
    parser.add_argument(
        "--npz_path",
        required=True,
        help="Path to synth_signals.npz",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Total number of signals to plot",
    )
    parser.add_argument(
        "--per_fig",
        type=int,
        default=10,
        help="Number of signals per figure",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for PNGs. Default: <npz_dir>/plots",
    )
    parser.add_argument(
        "--use_normalized",
        action="store_true",
        help="Use 'signals' instead of 'signals_raw' when both exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if args.use_normalized or "signals_raw" not in data.files:
        if "signals" not in data.files:
            raise KeyError("npz missing both 'signals_raw' and 'signals'")
        signals = data["signals"]
        tag = "norm"
    else:
        signals = data["signals_raw"]
        tag = "raw"

    if signals.ndim != 2:
        raise ValueError(f"expected 2D array (N, T), got shape={signals.shape}")

    n_total = min(max(args.num, 1), signals.shape[0])
    per_fig = max(1, args.per_fig)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_figs = (n_total + per_fig - 1) // per_fig
    for fig_idx in range(n_figs):
        start = fig_idx * per_fig
        end = min(n_total, start + per_fig)
        n_plot = end - start

        fig, axes = plt.subplots(n_plot, 1, figsize=(14, 2.0 * n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]

        for i in range(n_plot):
            idx = start + i
            axes[i].plot(signals[idx], lw=1.0)
            axes[i].set_ylabel(f"#{idx}")
            axes[i].grid(alpha=0.25)

        axes[-1].set_xlabel("Sample Index")
        fig.suptitle(f"Generated Signals ({tag}) {start}-{end - 1}", y=0.995)
        fig.tight_layout()
        out_path = out_dir / f"synth_signals_{tag}_{start:05d}_{end - 1:05d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
