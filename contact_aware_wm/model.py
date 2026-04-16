"""
CNN encoder-decoder world model for next-image prediction.

Two conditioning modes:
- "image_only": condition = [encoder(image), action]
- "image_ft":   condition = [encoder(image), action, ft]

Architecture is identical except for the conditioning MLP input size.
Input/output: (3, 96, 96) images.
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    CNN encoder-decoder world model.

    Encoder: (3, 96, 96) -> 256-dim latent
    Conditioning MLP: latent + action [+ ft] -> 256-dim
    Decoder: 256-dim -> (3, 96, 96) predicted next image
    """

    def __init__(self, condition="image_only"):
        super().__init__()
        assert condition in ("image_only", "image_ft")
        self.condition = condition

        latent_dim = 256
        action_dim = 3
        ft_dim = 6

        # Encoder: (3, 96, 96) -> spatial features
        # 96 -> 48 -> 24 -> 12 -> 6 with k=4, s=2, p=1
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),    # -> (32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # -> (64, 24, 24)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # -> (128, 12, 12)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # -> (256, 6, 6)
            nn.ReLU(),
        )
        # Flatten 256*6*6 = 9216 -> 256
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, latent_dim),
            nn.ReLU(),
        )

        # Conditioning MLP
        if condition == "image_only":
            cond_input_dim = latent_dim + action_dim        # 256 + 3 = 259
        else:
            cond_input_dim = latent_dim + action_dim + ft_dim  # 256 + 3 + 6 = 265

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Decoder: 256 -> (3, 96, 96)
        # 256 -> (256, 6, 6) -> upsample to (3, 96, 96)
        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 256 * 6 * 6),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> (128, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> (64, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> (32, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # -> (3, 96, 96)
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, image, action, ft=None):
        """
        Args:
            image:  (B, 3, 96, 96) float in [0, 1]
            action: (B, 3) float
            ft:     (B, 6) float (only used if condition == "image_ft")

        Returns:
            predicted_image: (B, 3, 96, 96) float in [0, 1]
        """
        z = self.encoder(image)
        z = self.encoder_fc(z)  # (B, 256)

        if self.condition == "image_only":
            cond = torch.cat([z, action], dim=1)
        else:
            assert ft is not None, "F/T required for image_ft mode"
            cond = torch.cat([z, action, ft], dim=1)

        cond = self.cond_mlp(cond)  # (B, 256)

        h = self.decoder_fc(cond)   # (B, 256*6*6)
        h = h.view(-1, 256, 6, 6)   # (B, 256, 6, 6)
        pred = self.decoder(h)       # (B, 3, 96, 96)
        return pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
