"""F/T encoder / decoder modules for Cosmos fine-tuning.

FTEncoder:  (B, T, 7) F/T chunk  ->  (B, d_model) single token
FTDecoder:  (B, d_model)          ->  (B, 6) predicted next F/T

Design notes
------------
- Input is 7D: 6 F/T channels + 1 binary contact indicator.
- We use a small temporal Conv1D stack so the encoder captures local
  dynamics (derivatives, contact transients) rather than just magnitude.
- AdaptiveAvgPool gives us a fixed-size chunk token regardless of T.
- Output dim matches Cosmos transformer hidden (2048 by default).
- Decoder is a thin MLP head; it predicts the 6D F/T at the future
  step from the transformer's output at the future_ft token position.
"""

import torch
import torch.nn as nn


def log_scale_ft(ft: torch.Tensor) -> torch.Tensor:
    """Log-scale F/T values to compress dynamic range.

    sign(f) * log(1 + |f|) keeps sign, compresses large contact spikes.
    Apply BEFORE z-score normalization.
    """
    return torch.sign(ft) * torch.log1p(ft.abs())


def add_contact_channel(ft: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Append a binary contact indicator channel.

    Args:
        ft: (B, T, 6) *normalized* F/T
        threshold: magnitude threshold (in normalized units) for contact

    Returns:
        (B, T, 7) — last channel is 1.0 where ||force|| > threshold else 0.0
    """
    # Use only force components (first 3) for contact detection
    force_mag = torch.linalg.norm(ft[..., :3], dim=-1, keepdim=True)
    contact = (force_mag > threshold).float()
    return torch.cat([ft, contact], dim=-1)


class FTEncoder(nn.Module):
    """Temporal Conv1D encoder that maps an F/T chunk to a single token.

    (B, T, in_dim) -> (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 128,
        out_dim: int = 2048,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, ft_chunk: torch.Tensor) -> torch.Tensor:
        # ft_chunk: (B, T, in_dim) -> (B, in_dim, T) for Conv1d
        x = ft_chunk.transpose(1, 2)
        x = self.conv(x)                      # (B, hidden, T)
        x = self.pool(x).squeeze(-1)          # (B, hidden)
        x = self.proj(x)                      # (B, out_dim)
        return x


class FTDecoder(nn.Module):
    """MLP head that predicts future F/T from a transformer output token.

    (B, in_dim) -> (B, 6)
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 128, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.net(token)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    # Quick shape + param-count sanity check
    B, T = 4, 16
    enc = FTEncoder(in_dim=7, out_dim=2048)
    dec = FTDecoder(in_dim=2048, out_dim=6)
    ft = torch.randn(B, T, 6)
    ft_scaled = log_scale_ft(ft)
    ft_with_contact = add_contact_channel(ft_scaled)
    token = enc(ft_with_contact)
    pred = dec(token)
    print(f"Input F/T chunk:     {ft.shape}")
    print(f"After log-scale:     {ft_scaled.shape}")
    print(f"After contact chan:  {ft_with_contact.shape}")
    print(f"Encoded token:       {token.shape}")
    print(f"Decoded F/T:         {pred.shape}")
    print(f"FTEncoder params:    {count_params(enc):,}")
    print(f"FTDecoder params:    {count_params(dec):,}")
