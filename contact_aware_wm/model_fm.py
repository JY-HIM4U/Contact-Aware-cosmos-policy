"""
Flow matching world model for next-image prediction.

Uses Conditional Optimal Transport (OT) flow matching:
  x_t = (1 - t) * x_0 + t * x_1,  where x_0 ~ N(0,I), x_1 = target image
  target velocity: dx_t = x_1 - x_0

Architecture:
  - Context encoder: shared CNN encodes N context frames → N × 256-dim
    with learned role embeddings (anchor vs memory positions)
  - Temporal encoder: 1D conv over action/ft chunks → 64-dim
  - Time embedding: sinusoidal → MLP → 256-dim
  - Velocity UNet2D: predicts velocity field v(x_t, t, cond) with FiLM conditioning
  - Output: velocity (B, 3, 96, 96) — unbounded

Supports:
  - context_frames=1 (backward compat): single frame, no role embeddings
  - context_frames=N with use_anchor: slot 0 has anchor role embedding,
    slots 1..N-1 have memory position embeddings
  - Noisy context augmentation: add N(0, noise_std) to memory frames during
    training to close the train-inference gap (World4RL / WoVR motivation)
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar timestep t in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TemporalEncoder(nn.Module):
    """Encode a temporal chunk of action or F/T into a fixed-size vector."""

    def __init__(self, in_dim, out_dim, max_horizon=16):
        super().__init__()
        hidden = max(out_dim, 64)
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool_fc = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.SiLU(),
        )

    def forward(self, x, mask=None):
        h = self.net(x.transpose(1, 2))
        h = h.transpose(1, 2)
        if mask is not None and mask.any():
            mask_f = mask.unsqueeze(-1).float()
            h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.pool_fc(h)


class FiLMResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, in_ch, out_ch, cond_dim, n_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(n_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(n_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        film_params = self.film(cond)
        scale, shift = film_params.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class FlowMatchingWorldModel(nn.Module):
    """
    Conditional flow matching world model using a UNet2D velocity network.

    Args:
        condition: "image_only" or "image_ft"
        context_frames: number of context frames (1 = backward compat, >1 = multi-frame)
        use_anchor: if True and context_frames > 1, slot 0 gets a distinct anchor
                    role embedding so the model learns to treat it as a stable reference
        ch_mult: UNet channel multipliers per resolution level
    """

    def __init__(self, condition="image_only", context_frames=1, use_anchor=False,
                 ch_mult=(64, 128, 256, 256)):
        super().__init__()
        assert condition in ("image_only", "image_ft")
        self.condition = condition
        self.context_frames = context_frames
        self.use_anchor = use_anchor and (context_frames > 1)

        latent_dim = 256
        action_dim = 3
        ft_dim = 6
        time_dim = 128
        action_embed_dim = 64
        ft_embed_dim = 64
        cond_dim = 256

        # --- Shared CNN encoder for each context frame ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 6 * 6, latent_dim), nn.ReLU(),
        )

        # --- Role embeddings for multi-frame context ---
        # Learned embeddings that distinguish anchor (slot 0) from memory positions.
        # This lets the model learn "trust slot 0 for static background" vs
        # "trust recent memory for dynamic objects" (World4RL motivation).
        if context_frames > 1:
            self.role_embeddings = nn.Parameter(torch.zeros(context_frames, latent_dim))
            nn.init.normal_(self.role_embeddings, std=0.02)

        # --- Context aggregation MLP ---
        # With N frames: N × 256-dim embeddings → 256-dim
        if context_frames > 1:
            self.context_agg = nn.Sequential(
                nn.Linear(context_frames * latent_dim, 512),
                nn.SiLU(),
                nn.Linear(512, latent_dim),
            )

        # --- Time embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Temporal encoders for action/ft chunks ---
        self.action_encoder = TemporalEncoder(action_dim, action_embed_dim)
        if condition == "image_ft":
            self.ft_encoder = TemporalEncoder(ft_dim, ft_embed_dim)

        # --- Final conditioning MLP ---
        if condition == "image_only":
            cond_input_dim = latent_dim + cond_dim + action_embed_dim
        else:
            cond_input_dim = latent_dim + cond_dim + action_embed_dim + ft_embed_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, cond_dim),
        )

        # --- Velocity UNet2D ---
        c0, c1, c2, c3 = ch_mult

        self.input_conv = nn.Conv2d(3, c0, 3, padding=1)

        self.down0 = FiLMResBlock(c0, c0, cond_dim)
        self.downsample0 = Downsample(c0)
        self.down1 = FiLMResBlock(c0, c1, cond_dim)
        self.downsample1 = Downsample(c1)
        self.down2 = FiLMResBlock(c1, c2, cond_dim)
        self.downsample2 = Downsample(c2)
        self.down3 = FiLMResBlock(c2, c3, cond_dim)
        self.downsample3 = Downsample(c3)

        self.mid = FiLMResBlock(c3, c3, cond_dim)

        self.upsample3 = Upsample(c3)
        self.up3 = FiLMResBlock(c3 + c3, c2, cond_dim)
        self.upsample2 = Upsample(c2)
        self.up2 = FiLMResBlock(c2 + c2, c1, cond_dim)
        self.upsample1 = Upsample(c1)
        self.up1 = FiLMResBlock(c1 + c1, c0, cond_dim)
        self.upsample0 = Upsample(c0)
        self.up0 = FiLMResBlock(c0 + c0, c0, cond_dim)

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, c0),
            nn.SiLU(),
            nn.Conv2d(c0, 3, 3, padding=1),
        )

    def _ensure_chunk(self, x):
        if x.ndim == 2:
            return x.unsqueeze(1)
        return x

    def _ensure_mask(self, mask, x):
        if mask is None:
            return torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        if mask.ndim == 1:
            return mask.unsqueeze(1)
        return mask

    def _encode_context(self, context_imgs):
        """Encode multi-frame context into a single latent vector.

        Args:
            context_imgs: (B, N, 3, 96, 96) multi-frame context
                          OR (B, 3, 96, 96) single frame (backward compat)

        Returns:
            z: (B, 256) aggregated context embedding
        """
        if context_imgs.ndim == 4:
            # Single frame (backward compat): (B, 3, 96, 96)
            return self.encoder_fc(self.encoder(context_imgs))

        B, N, C, H, W = context_imgs.shape

        # Encode each frame through the shared CNN
        # Reshape to (B*N, C, H, W) for batch processing
        flat = context_imgs.reshape(B * N, C, H, W)
        z_all = self.encoder_fc(self.encoder(flat))  # (B*N, 256)
        z_all = z_all.reshape(B, N, -1)              # (B, N, 256)

        # Add learned role embeddings to distinguish anchor from memory slots
        z_all = z_all + self.role_embeddings.unsqueeze(0)  # (B, N, 256)

        # Aggregate: concatenate all frame embeddings → MLP → 256
        z_flat = z_all.reshape(B, -1)                # (B, N*256)
        return self.context_agg(z_flat)              # (B, 256)

    def encode_condition(self, context_imgs, action, ft, t, pad_mask=None):
        """Encode all conditioning signals into a single vector.

        Args:
            context_imgs: (B, N, 3, 96, 96) or (B, 3, 96, 96)
            action:       (B, H, 3) or (B, 3)
            ft:           (B, H, 6) or (B, 6) or None
            t:            (B,) timestep
            pad_mask:     (B, H) or (B,) or None

        Returns:
            cond: (B, 256)
        """
        z = self._encode_context(context_imgs)
        t_emb = self.time_embed(t)

        action_chunk = self._ensure_chunk(action)
        action_mask = self._ensure_mask(pad_mask, action_chunk)
        action_emb = self.action_encoder(action_chunk, action_mask)

        if self.condition == "image_only":
            cond_in = torch.cat([z, t_emb, action_emb], dim=1)
        else:
            assert ft is not None
            ft_chunk = self._ensure_chunk(ft)
            ft_mask = self._ensure_mask(pad_mask, ft_chunk)
            ft_emb = self.ft_encoder(ft_chunk, ft_mask)
            cond_in = torch.cat([z, t_emb, action_emb, ft_emb], dim=1)

        return self.cond_mlp(cond_in)

    def forward(self, x_t, context_imgs, action, ft, t, pad_mask=None):
        """Predict the velocity field v(x_t, t, cond).

        Args:
            x_t:          (B, 3, 96, 96) noisy sample at time t
            context_imgs: (B, N, 3, 96, 96) or (B, 3, 96, 96)
            action:       (B, H, 3) or (B, 3)
            ft:           (B, H, 6) or (B, 6) or None
            t:            (B,) timestep in [0, 1]
            pad_mask:     (B, H) or None

        Returns:
            velocity: (B, 3, 96, 96)
        """
        cond = self.encode_condition(context_imgs, action, ft, t, pad_mask)

        h0 = self.input_conv(x_t)
        h0 = self.down0(h0, cond)
        h1 = self.downsample0(h0)
        h1 = self.down1(h1, cond)
        h2 = self.downsample1(h1)
        h2 = self.down2(h2, cond)
        h3 = self.downsample2(h2)
        h3 = self.down3(h3, cond)
        h4 = self.downsample3(h3)

        h4 = self.mid(h4, cond)

        h = self.upsample3(h4)
        h = self.up3(torch.cat([h, h3], dim=1), cond)
        h = self.upsample2(h)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        h = self.upsample0(h)
        h = self.up0(torch.cat([h, h0], dim=1), cond)

        return self.output_conv(h)

    @torch.no_grad()
    def sample(self, context_imgs, action, ft, num_steps=20, pad_mask=None,
               device=None, num_samples=1):
        """Generate next-image prediction via Euler ODE integration.

        Args:
            context_imgs: (B, N, 3, 96, 96) or (B, 3, 96, 96)
            action:       (B, H, 3) or (B, 3)
            ft:           (B, H, 6) or (B, 6) or None
            num_steps:    Euler integration steps
            pad_mask:     (B, H) or None
            num_samples:  number of noise samples to average

        Returns:
            predicted_image: (B, 3, 96, 96) clamped to [0, 1]
        """
        if device is None:
            device = context_imgs.device

        B = context_imgs.size(0)
        dt = 1.0 / num_steps

        accum = torch.zeros(B, 3, 96, 96, device=device)

        for _ in range(num_samples):
            x = torch.randn(B, 3, 96, 96, device=device)
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                v = self.forward(x, context_imgs, action, ft, t, pad_mask)
                x = x + dt * v
            accum += x

        return (accum / num_samples).clamp(0, 1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
