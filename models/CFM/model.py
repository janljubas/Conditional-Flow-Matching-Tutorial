"""
Velocity network architectures for Conditional Flow Matching.

CFM is a *training method* (Tong et al., 2024), not an architecture.
The neural network v_theta(x_t, t) that parameterizes the velocity field
can be any architecture that maps (x, t) -> velocity of the same shape as x.

This module provides:
  - VelocityNet        : abstract base class with shared CFM math helpers
  - ConvStackVelocityNet : the original dilated-conv backbone (formerly CFMModel)
  - UNetVelocityNet     : a U-Net backbone with skip connections, sinusoidal time
                          embedding, optional self-attention, and class conditioning
  - build_velocity_net  : factory function driven by config["backbone"]

Paper references (Tong et al., 2024 -- "Improving and Generalizing Flow-Based
Generative Models with Minibatch Optimal Transport"):
  - Eq.13: CFM loss  L_CFM = E[ ||v_theta(t, x_t) - u_t(x|z)||^2 ]
  - Eq.14: Probability path mean  mu_t = (1-t)*x_0 + t*x_1
  - Eq.15: Conditional vector field  u_t(x|x_0,x_1) = x_1 - x_0
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VelocityNet(nn.Module):
    """
    Abstract base for velocity networks used in CFM training.
    Subclasses must implement ``forward(x, t, class_labels=None) -> velocity``.
    Shared helpers: interpolate, get_velocity, sample.
    """

    def __init__(self, sigma_min: float = 0.0):
        super().__init__()
        self.sigma_min = sigma_min

    # -- CFM math helpers (shared across all backbones) ---------------------

    def get_velocity(self, x_0, x_1):
        """Conditional vector field u_t(x|x_0,x_1) = x_1 - (1 - sigma_min)*x_0.
        Tong et al. Eq.15 (reduces to x_1 - x_0 when sigma_min = 0)."""
        return x_1 - (1 - self.sigma_min) * x_0

    def interpolate(self, x_0, x_1, t):
        """Probability path mean mu_t = (1 - (1-sigma_min)*t)*x_0 + t*x_1.
        Tong et al. Eq.14 (reduces to (1-t)*x_0 + t*x_1 when sigma_min = 0).
        With sigma ~ 0 the conditional distribution is a Dirac delta at mu_t."""
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    @torch.no_grad()
    def sample(self, t_steps, shape, device, noise_sampler=None, class_labels=None):
        """Generate samples by Euler-integrating the learned ODE
        dx/dt = v_theta(x, t) from t=0 to t=1.

        If noise_sampler is provided, initial x_0 is drawn from it (e.g. quantum);
        otherwise falls back to standard Gaussian.
        class_labels: optional (B,) int tensor for class-conditional generation."""
        if noise_sampler is not None:
            x = noise_sampler.sample(shape, device)
        else:
            x = torch.randn(size=shape, device=device)
        delta = 1.0 / max(t_steps - 1, 1)
        t_vals = torch.linspace(0, 1, t_steps, device=device)

        for i in range(t_steps - 1):
            t_cur = t_vals[i].view(1, 1, 1, 1).expand(shape[0], 1, 1, 1)
            x = x + self(x, t_cur, class_labels=class_labels) * delta
        return x

    def forward(self, x, t, class_labels=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ConvBlock (shared building block)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Conv2d):
    """Conv2d with optional GroupNorm + SiLU, supporting residual time-injection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation_fn=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
        gn=False,
        gn_groups=8,
    ):
        if padding == "same":
            padding = kernel_size // 2 * dilation

        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None
        self.activation_fn = nn.SiLU() if activation_fn else None

    def forward(self, x, time_embedding=None, residual=False):
        if residual:
            x = x + time_embedding
            y = x
            x = super().forward(x)
            y = y + x
        else:
            y = super().forward(x)
        if self.group_norm is not None:
            y = self.group_norm(y)
        if self.activation_fn is not None:
            y = self.activation_fn(y)
        return y


# ---------------------------------------------------------------------------
# Backbone 1: ConvStackVelocityNet (the original architecture)
# ---------------------------------------------------------------------------

class ConvStackVelocityNet(VelocityNet):
    """Dilated conv-stack velocity network (formerly CFMModel).
    Grows receptive field exponentially via dilation without downsampling."""

    def __init__(self, image_resolution, hidden_dims=None, sigma_min=0.0):
        super().__init__(sigma_min=sigma_min)
        if hidden_dims is None:
            hidden_dims = [256, 256]

        _, _, img_c = image_resolution
        self.in_project = ConvBlock(img_c, hidden_dims[0], kernel_size=7)
        self.time_project = nn.Sequential(
            ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[-1], kernel_size=1),
        )

        self.convs = nn.ModuleList(
            [ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=3)]
        )
        for idx in range(1, len(hidden_dims)):
            self.convs.append(
                ConvBlock(
                    hidden_dims[idx - 1], hidden_dims[idx], kernel_size=3,
                    dilation=3 ** ((idx - 1) // 2),
                    activation_fn=True, gn=True, gn_groups=8,
                )
            )

        self.out_project = ConvBlock(hidden_dims[-1], img_c, kernel_size=3)

    def forward(self, x, t, class_labels=None):
        te = self.time_project(t)
        y = self.in_project(x)
        for block in self.convs:
            y = block(y, te, residual=True)
        return self.out_project(y)


# ---------------------------------------------------------------------------
# Backbone 2: UNetVelocityNet
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar t, followed by a 2-layer MLP.
    Maps t of shape (B,1,1,1) -> (B, dim)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        t_flat = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t_flat[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial dimensions with residual connection.
    Follows the approach from guided-diffusion / DDPM architectures."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H * W)
        out = self.proj(out).view(B, C, H, W)
        return x + out


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, Conv, and additive time-conditioning."""

    def __init__(self, channels, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class DownBlock(nn.Module):
    """Two ResBlocks + optional self-attention + stride-2 downsampling conv."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0, use_attention=False, num_heads=4):
        super().__init__()
        self.res1 = ResBlock(in_ch, time_dim, dropout)
        self.res2 = ResBlock(in_ch, time_dim, dropout)
        self.attn = SelfAttention(in_ch, num_heads) if use_attention else nn.Identity()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        h = self.attn(h)
        skip = h
        h = self.down(h)
        return h, skip


class UpBlock(nn.Module):
    """Upsample + concat skip + two ResBlocks + optional self-attention."""

    def __init__(self, in_ch, skip_ch, out_ch, time_dim, dropout=0.0, use_attention=False, num_heads=4):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        merged = in_ch + skip_ch
        self.conv_merge = nn.Conv2d(merged, out_ch, 1)
        self.res1 = ResBlock(out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, time_dim, dropout)
        self.attn = SelfAttention(out_ch, num_heads) if use_attention else nn.Identity()

    def forward(self, x, skip, t_emb):
        h = self.up(x)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = torch.cat([h, skip], dim=1)
        h = self.conv_merge(h)
        h = self.res1(h, t_emb)
        h = self.res2(h, t_emb)
        h = self.attn(h)
        return h


class UNetVelocityNet(VelocityNet):
    """
    U-Net velocity network for CFM with optional self-attention and class conditioning.

    Architecture (for 28x28 MNIST or 32x32 CIFAR):
        Encoder:  [C,H,W] -> [base,H,W] -> [base*2,H/2,W/2] -> [base*4,H/4,W/4]
        Bottleneck: ResBlock -> SelfAttention -> ResBlock at lowest resolution
        Decoder:  mirrors encoder with skip connections
        Output:   [C,H,W] (predicted velocity, same shape as input)

    Time conditioning: sinusoidal embedding -> MLP -> additive bias at every ResBlock.
    Class conditioning: nn.Embedding(num_classes, time_dim) added to time embedding.
    """

    def __init__(self, image_resolution, base_channels=64, sigma_min=0.0,
                 dropout=0.0, num_classes=0, use_attention=True, num_heads=4):
        super().__init__(sigma_min=sigma_min)
        _, _, img_c = image_resolution
        time_dim = base_channels * 4
        ch1, ch2, ch3 = base_channels, base_channels * 2, base_channels * 4

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        self.class_embed = nn.Embedding(num_classes, time_dim) if num_classes > 0 else None

        self.conv_in = nn.Conv2d(img_c, ch1, 3, padding=1)

        self.down1 = DownBlock(ch1, ch2, time_dim, dropout)
        self.down2 = DownBlock(ch2, ch3, time_dim, dropout)

        self.bottleneck1 = ResBlock(ch3, time_dim, dropout)
        self.bottleneck_attn = SelfAttention(ch3, num_heads) if use_attention else nn.Identity()
        self.bottleneck2 = ResBlock(ch3, time_dim, dropout)

        self.up1 = UpBlock(ch3, ch2, ch2, time_dim, dropout)
        self.up2 = UpBlock(ch2, ch1, ch1, time_dim, dropout)

        self.norm_out = nn.GroupNorm(min(8, ch1), ch1)
        self.conv_out = nn.Conv2d(ch1, img_c, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t, class_labels=None):
        t_emb = self.time_embed(t)

        if class_labels is not None and self.class_embed is not None:
            t_emb = t_emb + self.class_embed(class_labels)

        h = self.conv_in(x)

        h, skip1 = self.down1(h, t_emb)
        h, skip2 = self.down2(h, t_emb)

        h = self.bottleneck1(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck2(h, t_emb)

        h = self.up1(h, skip2, t_emb)
        h = self.up2(h, skip1, t_emb)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Backward compat alias
# ---------------------------------------------------------------------------
CFMModel = ConvStackVelocityNet


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_velocity_net(settings):
    """Instantiate a velocity network based on settings['backbone']."""
    backbone = settings.get("backbone", "convstack")
    sigma_min = settings.get("sigma_min", 0.0)
    img_res = settings["img_size"]

    if backbone == "convstack":
        return ConvStackVelocityNet(
            image_resolution=img_res,
            hidden_dims=settings.get("hidden_dims"),
            sigma_min=sigma_min,
        )
    elif backbone == "unet":
        return UNetVelocityNet(
            image_resolution=img_res,
            base_channels=settings.get("unet_base_channels", 64),
            sigma_min=sigma_min,
            dropout=settings.get("dropout", 0.0),
            num_classes=settings.get("num_classes", 0),
            use_attention=settings.get("use_attention", True),
            num_heads=settings.get("num_heads", 4),
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'convstack' or 'unet'.")
