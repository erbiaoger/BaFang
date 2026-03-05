import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np

from diffusion_dataset import (
    build_signal_matrix,
    load_records_from_dir,
    normalize_signals,
    save_stats_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1D DDPM for vehicle signal generation")
    parser.add_argument("--pkl_dir", required=True, help="Directory containing grouped vehicle .pkl files")
    parser.add_argument("--out_dir", required=True, help="Output directory for checkpoints and config")
    parser.add_argument("--sample_length", type=int, default=None, help="Target signal length; default auto-infer")
    parser.add_argument("--length_mode", choices=["filter", "crop", "pad"], default="filter")
    parser.add_argument("--normalize", choices=["per_sample", "global"], default="per_sample")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", choices=["none", "cosine", "step", "exp"], default="cosine")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine schedule")
    parser.add_argument("--step_size", type=int, default=20, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for StepLR/ExponentialLR")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--ema", dest="ema", action="store_true", help="Enable EMA (default)")
    ema_group.add_argument("--no_ema", dest="ema", action="store_false", help="Disable EMA")
    parser.set_defaults(ema=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--auto_resume", action="store_true", help="Auto-resume from ckpt_last.pt or ckpt_best.pt")
    parser.add_argument("--resume_path", default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--resume_from", choices=["last", "best"], default="last", help="Auto-resume preference")
    return parser.parse_args()


class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def copy_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.shadow = state.get("shadow", self.shadow)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_config(args: argparse.Namespace, data_info: Dict, stats: Dict, total_samples: int) -> Dict:
    return {
        "args": vars(args),
        "dataset": data_info,
        "normalization": stats,
        "total_samples": total_samples,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install with: pip install torch") from exc

    from diffusion_model import GaussianDiffusion1D, UNet1D

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    records = load_records_from_dir(args.pkl_dir)
    signals, metas, data_info = build_signal_matrix(
        records,
        sample_length=args.sample_length,
        length_mode=args.length_mode,
    )
    signals_norm, stats = normalize_signals(signals, mode=args.normalize)

    n = signals_norm.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_n = max(1, int(n * args.val_ratio))
    train_idx = indices[val_n:]
    val_idx = indices[:val_n]

    x_train = torch.from_numpy(signals_norm[train_idx]).unsqueeze(1)
    x_val = torch.from_numpy(signals_norm[val_idx]).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(x_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = UNet1D(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion1D(timesteps=args.timesteps, beta_schedule=args.beta_schedule).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_schedule == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    best_val = float("inf")
    start_epoch = 1

    resume_ckpt = None
    if args.resume_path:
        resume_ckpt = Path(args.resume_path)
    elif args.auto_resume:
        last_path = out_dir / "ckpt_last.pt"
        best_path = out_dir / "ckpt_best.pt"
        if args.resume_from == "last" and last_path.exists():
            resume_ckpt = last_path
        elif args.resume_from == "best" and best_path.exists():
            resume_ckpt = best_path
        elif last_path.exists():
            resume_ckpt = last_path
        elif best_path.exists():
            resume_ckpt = best_path

    if resume_ckpt is not None and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "val_loss" in ckpt:
            best_val = min(best_val, float(ckpt["val_loss"]))
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {resume_ckpt} at epoch {start_epoch}")

    save_stats_json(out_dir / "stats.json", stats)
    config = build_config(args, data_info, stats, total_samples=n)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            t = torch.randint(0, diffusion.timesteps, (xb.size(0),), device=device).long()
            loss = diffusion.p_losses(model, xb, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                t = torch.randint(0, diffusion.timesteps, (xb.size(0),), device=device).long()
                val_loss = diffusion.p_losses(model, xb, t)
                val_losses.append(val_loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.6g}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
            "sample_length": int(signals.shape[1]),
            "real_num_samples": int(n),
        }

        if ema is not None:
            ema_model = UNet1D(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
            ema.copy_to(ema_model)
            ckpt["ema_model"] = ema_model.state_dict()
            ckpt["ema"] = ema.state_dict()
            del ema_model

        if epoch % args.save_every == 0:
            torch.save(ckpt, out_dir / f"ckpt_{epoch:04d}.pt")
        torch.save(ckpt, out_dir / "ckpt_last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "ckpt_best.pt")

        if scheduler is not None:
            scheduler.step()

    print(f"Training done. Best val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()




# python /Users/zhangzhiyu/MyProjects/BaFang/diffusion/diffusion_train.py \
#   --pkl_dir /Users/zhangzhiyu/MyProjects/BaFang/data/peaks_agc \
#   --out_dir /Users/zhangzhiyu/MyProjects/BaFang/data/ddpm_out_agc \
#   --normalize per_sample \
#   --timesteps 1000 \
#   --batch_size 64 \
#   --epochs 100 \
#   --lr 1e-4
