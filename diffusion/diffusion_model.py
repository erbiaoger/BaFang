import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.deconv(x)
        if y.shape[-1] != target_len:
            y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
        return y


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        dims = [base_channels * m for m in channel_mults]

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev = base_channels
        for i, d in enumerate(dims):
            self.down_blocks.append(ResBlock1D(prev, d, time_dim))
            if i < len(dims) - 1:
                self.downsamples.append(Downsample1D(d))
            prev = d

        self.mid1 = ResBlock1D(prev, prev, time_dim)
        self.mid2 = ResBlock1D(prev, prev, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        rev_dims = list(reversed(dims))
        for i, d in enumerate(rev_dims):
            in_ch = prev + d
            self.up_blocks.append(ResBlock1D(in_ch, d, time_dim))
            prev = d
            if i < len(rev_dims) - 1:
                self.upsamples.append(Upsample1D(prev))

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(prev, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.in_conv(x)
        skips = []
        sizes = []

        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            skips.append(h)
            sizes.append(h.shape[-1])
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            if i < len(self.upsamples):
                target_len = sizes[-(i + 2)]
                h = self.upsamples[i](h, target_len)

        return self.out_conv(self.out_act(self.out_norm(h)))


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999).float()


class GaussianDiffusion1D(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine"):
        super().__init__()
        self.timesteps = timesteps

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"unsupported beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def p_losses(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, step: int) -> torch.Tensor:
        beta_t = _extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - beta_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if step == 0:
            return model_mean

        posterior_variance_t = _extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        b = shape[0]
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, step)
        return x
