"""
Improved flow matching world model for LIBERO.

Key fixes over model_fm.py (the baseline):
  1. Wider bottleneck: frame_embed=512, cond_dim=512 (was 256)
  2. Separate F/T FiLM path: F/T no longer competes with visual features
     in the conditioning bottleneck — each UNet block has dual FiLM layers
  3. Spatial context injection: anchor frame's spatial features are concatenated
     at high-resolution UNet levels (96×96 and 48×48), preserving WHERE objects are
  4. Wider action/FT embeddings: 128-dim (was 64)

Architecture:
  Frame Encoder: (B, 3, 96, 96) → (B, 512)
  Context Agg:   (B, N, 512) + role_emb → MLP → (B, 512)
  Time Emb:      scalar → (B, 512)
  Action Enc:    (B, H, 7) → (B, 128)
  Visual Cond:   cat[ctx(512), time(512), action(128)] → MLP → (B, 512)
  FT Cond:       (B, H, 6) → TemporalEnc(128) → MLP → (B, 256)  [SEPARATE]
  UNet:          Each block has FiLM_visual + FiLM_ft (dual modulation)
  Spatial Inj:   Anchor CNN features at 96×96 and 48×48 concatenated to UNet encoder
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for the improved world model."""
    condition: str = "image_ft"       # "image_only" or "image_ft"
    context_frames: int = 4
    use_anchor: bool = True
    use_ft: bool = True
    img_size: int = 96
    action_dim: int = 7               # LIBERO: 7D actions
    ft_dim: int = 6

    # Encoder dims — wider than baseline
    frame_embed_dim: int = 512        # was 256
    context_agg_dim: int = 512        # was 256
    action_embed_dim: int = 128       # was 64
    ft_embed_dim: int = 128           # was 64
    t_embed_dim: int = 512            # was 256
    cond_dim: int = 512               # was 256 — final visual conditioning dim
    ft_cond_dim: int = 256            # separate F/T conditioning dim

    # UNet dims — doubled base channels
    unet_ch_mult: Tuple[int, ...] = (128, 256, 512, 512)

    # Perceptual loss weight (0 = MSE only, >0 = MSE + LPIPS)
    lpips_weight: float = 0.0


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TemporalEncoder(nn.Module):
    """Encode action/FT chunks via 1D conv + masked mean pool."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(out_dim, 64)
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool_fc = nn.Sequential(nn.Linear(hidden, out_dim), nn.SiLU())

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None and mask.any():
            m = mask.unsqueeze(-1).float()
            h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.pool_fc(h)


class DualFiLMResBlock(nn.Module):
    """Residual block with DUAL FiLM conditioning: visual + F/T.

    Visual FiLM modulates after the first GroupNorm (coarse, scene-level).
    F/T FiLM modulates after the second GroupNorm (fine, contact-level).
    This prevents F/T from competing with visual features for representation
    capacity in the conditioning bottleneck.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 ft_cond_dim: int = 0, n_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(n_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(n_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

        # Visual FiLM (always present)
        self.film_visual = nn.Linear(cond_dim, out_ch * 2)

        # F/T FiLM (separate pathway, optional)
        self.has_ft = ft_cond_dim > 0
        if self.has_ft:
            self.film_ft = nn.Linear(ft_cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, visual_cond: torch.Tensor,
                ft_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Visual FiLM modulation
        scale_v, shift_v = self.film_visual(visual_cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale_v[:, :, None, None]) + shift_v[:, :, None, None]

        # F/T FiLM modulation (separate, additive)
        if self.has_ft and ft_cond is not None:
            scale_f, shift_f = self.film_ft(ft_cond).chunk(2, dim=1)
            h = h * (1 + scale_f[:, :, None, None]) + shift_f[:, :, None, None]

        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ImprovedWorldModel(nn.Module):
    """Improved flow matching world model with separate F/T path and spatial injection.

    Supports both LIBERO (7D actions) and RH20T (3D actions) via config.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.use_ft = config.use_ft
        self.img_size = config.img_size

        # Spatial dims after 4 stride-2 convs: img_size / 16
        # 96 → 6, 128 → 8
        enc_spatial = config.img_size // 16

        # --- Frame Encoder: wider (512-dim) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),    # S→S/2
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),  # S/2→S/4
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(), # S/4→S/8
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU(), # S/8→S/16
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * enc_spatial * enc_spatial, config.frame_embed_dim),
            nn.ReLU(),
        )

        # Spatial feature taps for anchor injection (pre-downsample features)
        # Level 0: after first conv (64ch, 48×48)
        # Level 1: after second conv (128ch, 24×24)
        # We'll extract these during forward pass

        # Spatial injection projections: project anchor features to UNet channel count
        c0, c1, c2, c3 = config.unet_ch_mult
        self.spatial_proj_0 = nn.Conv2d(64, c0, 1)   # 64→c0 at 48×48
        self.spatial_proj_1 = nn.Conv2d(128, c1, 1)   # 128→c1 at 24×24

        # --- Role embeddings for multi-frame context ---
        if config.context_frames > 1:
            self.role_embeddings = nn.Parameter(
                torch.zeros(config.context_frames, config.frame_embed_dim))
            nn.init.normal_(self.role_embeddings, std=0.02)

            self.context_agg = nn.Sequential(
                nn.LayerNorm(config.context_frames * config.frame_embed_dim),
                nn.Linear(config.context_frames * config.frame_embed_dim, 1024),
                nn.SiLU(),
                nn.Linear(1024, config.context_agg_dim),
            )

        # --- Time embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.t_embed_dim // 2),
            nn.Linear(config.t_embed_dim // 2, config.t_embed_dim),
            nn.SiLU(),
            nn.Linear(config.t_embed_dim, config.t_embed_dim),
        )

        # --- Action temporal encoder ---
        self.action_encoder = TemporalEncoder(config.action_dim, config.action_embed_dim)

        # --- Visual conditioning MLP (no F/T here) ---
        vis_input_dim = config.context_agg_dim + config.t_embed_dim + config.action_embed_dim
        self.vis_cond_mlp = nn.Sequential(
            nn.Linear(vis_input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, config.cond_dim),
        )

        # --- Separate F/T conditioning path ---
        if config.use_ft:
            self.ft_encoder = TemporalEncoder(config.ft_dim, config.ft_embed_dim)
            self.ft_cond_mlp = nn.Sequential(
                nn.Linear(config.ft_embed_dim, 256),
                nn.SiLU(),
                nn.Linear(256, config.ft_cond_dim),
            )

        ft_cd = config.ft_cond_dim if config.use_ft else 0

        # --- Velocity UNet2D with dual FiLM ---
        # Input conv: 3 channels (noisy image)
        self.input_conv = nn.Conv2d(3, c0, 3, padding=1)

        # Encoder path — spatial injection at levels 0 and 1
        # Level 0: after spatial injection, in_ch = c0 + c0 (UNet + anchor spatial)
        self.down0 = DualFiLMResBlock(c0 + c0, c0, config.cond_dim, ft_cd)
        self.downsample0 = Downsample(c0)   # 96→48

        # Level 1: after spatial injection, in_ch = c0 + c1
        self.down1 = DualFiLMResBlock(c0 + c1, c1, config.cond_dim, ft_cd)
        self.downsample1 = Downsample(c1)   # 48→24

        # Levels 2-3: no spatial injection
        self.down2 = DualFiLMResBlock(c1, c2, config.cond_dim, ft_cd)
        self.downsample2 = Downsample(c2)   # 24→12
        self.down3 = DualFiLMResBlock(c2, c3, config.cond_dim, ft_cd)
        self.downsample3 = Downsample(c3)   # 12→6

        # Bottleneck
        self.mid = DualFiLMResBlock(c3, c3, config.cond_dim, ft_cd)

        # Decoder path (skip connections)
        self.upsample3 = Upsample(c3)
        self.up3 = DualFiLMResBlock(c3 + c3, c2, config.cond_dim, ft_cd)
        self.upsample2 = Upsample(c2)
        self.up2 = DualFiLMResBlock(c2 + c2, c1, config.cond_dim, ft_cd)
        self.upsample1 = Upsample(c1)
        self.up1 = DualFiLMResBlock(c1 + c1, c0, config.cond_dim, ft_cd)
        self.upsample0 = Upsample(c0)
        self.up0 = DualFiLMResBlock(c0 + c0, c0, config.cond_dim, ft_cd)

        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, c0),
            nn.SiLU(),
            nn.Conv2d(c0, 3, 3, padding=1),
        )

    def _ensure_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.ndim == 2 else x

    def _ensure_mask(self, mask: Optional[torch.Tensor],
                     x: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        if mask.ndim == 1:
            return mask.unsqueeze(1)
        return mask

    def _encode_anchor_spatial(self, anchor: torch.Tensor):
        """Extract spatial features from anchor frame at multiple resolutions.

        Args:
            anchor: (B, 3, 96, 96)

        Returns:
            spatial_48: (B, c0, 48, 48) — projected anchor features
            spatial_24: (B, c1, 24, 24)
        """
        # Run through encoder layers one by one to tap spatial features
        x = self.encoder[0](anchor)  # Conv2d(3→64), stride=2 → (B, 64, 48, 48)
        x = self.encoder[1](x)       # ReLU
        spatial_48 = self.spatial_proj_0(x)  # → (B, c0, 48, 48)

        x = self.encoder[2](x)       # Conv2d(64→128), stride=2 → (B, 128, 24, 24)
        x = self.encoder[3](x)       # ReLU
        spatial_24 = self.spatial_proj_1(x)  # → (B, c1, 24, 24)

        return spatial_48, spatial_24

    def _encode_context(self, context_imgs: torch.Tensor) -> torch.Tensor:
        """Encode multi-frame context into aggregated vector.

        Args:
            context_imgs: (B, N, 3, 96, 96) or (B, 3, 96, 96)

        Returns:
            z: (B, context_agg_dim)
        """
        if context_imgs.ndim == 4:
            return self.encoder_fc(self.encoder(context_imgs))

        B, N, C, H, W = context_imgs.shape
        flat = context_imgs.reshape(B * N, C, H, W)
        z_all = self.encoder_fc(self.encoder(flat)).reshape(B, N, -1)

        z_all = z_all + self.role_embeddings.unsqueeze(0)
        return self.context_agg(z_all.reshape(B, -1))

    def encode_condition(self, context_imgs: torch.Tensor,
                         action: torch.Tensor,
                         ft: Optional[torch.Tensor],
                         t: torch.Tensor,
                         pad_mask: Optional[torch.Tensor] = None):
        """Encode visual and F/T conditioning separately.

        Returns:
            visual_cond: (B, cond_dim)
            ft_cond: (B, ft_cond_dim) or None
        """
        z = self._encode_context(context_imgs)
        t_emb = self.time_embed(t)

        action_chunk = self._ensure_chunk(action)
        action_mask = self._ensure_mask(pad_mask, action_chunk)
        action_emb = self.action_encoder(action_chunk, action_mask)

        visual_cond = self.vis_cond_mlp(torch.cat([z, t_emb, action_emb], dim=1))

        ft_cond = None
        if self.use_ft and ft is not None:
            ft_chunk = self._ensure_chunk(ft)
            ft_mask = self._ensure_mask(pad_mask, ft_chunk)
            ft_emb = self.ft_encoder(ft_chunk, ft_mask)
            ft_cond = self.ft_cond_mlp(ft_emb)

        return visual_cond, ft_cond

    def forward(self, x_t: torch.Tensor, context_imgs: torch.Tensor,
                action: torch.Tensor, ft: Optional[torch.Tensor],
                t: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                anchor_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict velocity field v(x_t, t, cond).

        Args:
            x_t:          (B, 3, 96, 96) noisy sample
            context_imgs: (B, N, 3, 96, 96) or (B, 3, 96, 96)
            action:       (B, H, 7) or (B, 7)
            ft:           (B, H, 6) or None
            t:            (B,) timestep
            pad_mask:     (B, H) or None
            anchor_frame: (B, 3, 96, 96) — for spatial injection. If None, uses
                          context_imgs[:, 0] when multi-frame, or context_imgs itself.

        Returns:
            velocity: (B, 3, 96, 96)
        """
        visual_cond, ft_cond = self.encode_condition(context_imgs, action, ft, t, pad_mask)

        # Get anchor for spatial injection
        if anchor_frame is None:
            if context_imgs.ndim == 5:
                anchor_frame = context_imgs[:, 0]
            else:
                anchor_frame = context_imgs
        spatial_48, spatial_24 = self._encode_anchor_spatial(anchor_frame)

        # UNet encoder with spatial injection at levels 0 and 1
        S = self.img_size
        h0 = self.input_conv(x_t)                              # (B, c0, S, S)

        # Level 0: upsample anchor spatial features to full resolution
        sp_full = torch.nn.functional.interpolate(
            spatial_48, size=(S, S), mode='bilinear', align_corners=False)
        h0 = torch.cat([h0, sp_full], dim=1)                   # (B, c0+c0, S, S)
        h0 = self.down0(h0, visual_cond, ft_cond)              # (B, c0, S, S)
        h1 = self.downsample0(h0)                               # (B, c0, S/2, S/2)

        # Level 1: upsample anchor spatial features to half resolution
        sp_half = torch.nn.functional.interpolate(
            spatial_24, size=(S // 2, S // 2), mode='bilinear', align_corners=False)
        h1 = torch.cat([h1, sp_half], dim=1)                   # (B, c0+c1, S/2, S/2)
        h1 = self.down1(h1, visual_cond, ft_cond)              # (B, c1, 48, 48)
        h2 = self.downsample1(h1)                               # (B, c1, 24, 24)

        # Levels 2-3: no spatial injection
        h2 = self.down2(h2, visual_cond, ft_cond)              # (B, c2, 24, 24)
        h3 = self.downsample2(h2)                               # (B, c2, 12, 12)
        h3 = self.down3(h3, visual_cond, ft_cond)              # (B, c3, 12, 12)
        h4 = self.downsample3(h3)                               # (B, c3, 6, 6)

        # Bottleneck
        h4 = self.mid(h4, visual_cond, ft_cond)

        # Decoder
        h = self.upsample3(h4)
        h = self.up3(torch.cat([h, h3], dim=1), visual_cond, ft_cond)
        h = self.upsample2(h)
        h = self.up2(torch.cat([h, h2], dim=1), visual_cond, ft_cond)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h1], dim=1), visual_cond, ft_cond)
        h = self.upsample0(h)
        h = self.up0(torch.cat([h, h0], dim=1), visual_cond, ft_cond)

        return self.output_conv(h)

    @torch.no_grad()
    def sample(self, context_imgs: torch.Tensor, action: torch.Tensor,
               ft: Optional[torch.Tensor], num_steps: int = 50,
               pad_mask: Optional[torch.Tensor] = None,
               anchor_frame: Optional[torch.Tensor] = None,
               num_samples: int = 1) -> torch.Tensor:
        """Generate via Euler ODE integration.

        Defaults changed for sharper output:
          num_steps=50 (was 20) — more steps = better ODE approximation
          num_samples=1 (was variable) — single sample avoids blurry averaging
        Multi-sample averaging destroys high-frequency detail because each
        sample predicts a slightly different plausible future.
        """
        device = context_imgs.device
        B = context_imgs.size(0)
        S = self.img_size
        dt = 1.0 / num_steps

        accum = torch.zeros(B, 3, S, S, device=device)
        for _ in range(num_samples):
            x = torch.randn(B, 3, S, S, device=device)
            for i in range(num_steps):
                t_val = torch.full((B,), i * dt, device=device)
                v = self.forward(x, context_imgs, action, ft, t_val, pad_mask, anchor_frame)
                x = x + dt * v
            accum += x

        return (accum / num_samples).clamp(0, 1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
