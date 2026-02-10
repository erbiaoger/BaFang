import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from diffusion_dataset import denormalize_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample synthetic vehicle signals with trained 1D DDPM")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--num", type=int, default=None, help="Total number of samples to generate")
    parser.add_argument("--gen_multiplier", type=float, default=3.0, help="Generated count = real_count * multiplier")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use_ema", dest="use_ema", action="store_true", help="Use EMA weights (default)")
    ema_group.add_argument("--no_ema", dest="use_ema", action="store_false", help="Use raw model weights")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--save_grouped_pkl", action="store_true", help="Also save grouped-compatible pkl")
    return parser.parse_args()


def save_grouped_compatible(signals: np.ndarray, out_path: Path) -> None:
    grouped = {}
    for i in range(signals.shape[0]):
        grouped[i] = {
            "value": [signals[i]],
            "station": [-1],
            "sta_name": ["SYNTH"],
            "time": [-1],
            "veh_id": i,
            "fs": None,
        }
    with open(out_path, "wb") as f:
        pickle.dump(grouped, f)


def main() -> None:
    args = parse_args()

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install with: pip install torch") from exc

    from diffusion_model import GaussianDiffusion1D, UNet1D

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    cfg = ckpt.get("config", {})
    model_args = cfg.get("args", {})
    base_channels = int(model_args.get("base_channels", 64))
    timesteps = int(model_args.get("timesteps", 1000))
    beta_schedule = model_args.get("beta_schedule", "cosine")

    model = UNet1D(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    if args.use_ema and "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion1D(timesteps=timesteps, beta_schedule=beta_schedule).to(device)

    real_n = int(ckpt.get("real_num_samples", cfg.get("total_samples", 0)))
    target_num = args.num if args.num is not None else int(max(1, round(real_n * args.gen_multiplier)))
    sample_length = int(ckpt.get("sample_length", cfg.get("dataset", {}).get("target_length", 0)))
    if sample_length <= 0:
        raise ValueError("sample_length missing in checkpoint/config")

    chunks = []
    remaining = target_num
    while remaining > 0:
        b = min(args.batch_size, remaining)
        with torch.no_grad():
            x = diffusion.sample(model, (b, 1, sample_length), device=device)
        chunks.append(x.squeeze(1).cpu().numpy().astype(np.float32))
        remaining -= b

    synth = np.concatenate(chunks, axis=0)

    norm_stats = cfg.get("normalization", {"mode": "per_sample"})
    if norm_stats.get("mode") == "global":
        synth_raw = denormalize_signals(synth, norm_stats).astype(np.float32)
    else:
        synth_raw = synth

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint": str(args.ckpt),
        "num_generated": int(target_num),
        "sample_length": int(sample_length),
        "normalization": norm_stats,
    }

    npz_path = out_dir / "synth_signals.npz"
    np.savez(npz_path, signals=synth.astype(np.float32), signals_raw=synth_raw.astype(np.float32), meta=np.array(meta, dtype=object))

    if args.save_grouped_pkl:
        save_grouped_compatible(synth_raw, out_dir / "synth_grouped.pkl")

    print(f"Saved synthetic samples: {npz_path}")


if __name__ == "__main__":
    main()
