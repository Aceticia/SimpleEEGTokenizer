from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ


def get_activation(dim, activation_type):
    if activation_type == "snake":
        return SnakeActivation(dim)
    elif activation_type == "gelu":
        return nn.GELU()
    else:
        return nn.ReLU()


class SnakeActivation(nn.Module):
    """
    Snake activation function with learnable parameters: x + (1/a) * sin^2(a*x)
    Paper: https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features: int, alpha: float = 1.0):
        super().__init__()
        # Learnable frequency parameter per feature
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

    def forward(self, x):
        return x + (1 / self.alpha) * torch.sin(self.alpha * x) ** 2


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer for conditioning
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.scale_net = nn.Linear(condition_dim, feature_dim)
        self.shift_net = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, condition):
        """
        Apply FiLM conditioning

        Args:
            x: Features of shape [batch_size, feature_dim]
            condition: Conditioning vector of shape [batch_size, condition_dim]

        Returns:
            Conditioned features
        """
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)
        return x * (1 + scale) + shift


class ResidualMLPBlock(nn.Module):
    """
    A modular MLP block with optional residual connections, normalization,
    dropout, and FiLM conditioning.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_name: str = "relu",
        dropout: float = 0.0,
        norm_type: str = None,
        use_residual: bool = False,
        use_gelu_gating: bool = False,
        use_film: bool = False,
        condition_dim: int = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual and (
            in_dim == out_dim
        )  # Only residual if dims match
        self.use_film = use_film
        self.use_gelu_gating = use_gelu_gating

        # Main transformation
        if use_gelu_gating:
            # For gating, we need two linear layers (gate and value)
            self.linear_gate = nn.Linear(in_dim, out_dim)
            self.linear_value = nn.Linear(in_dim, out_dim)
        else:
            self.linear = nn.Linear(in_dim, out_dim)

        # Normalization
        if norm_type == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm_type == "rms":
            self.norm = nn.RMSNorm(out_dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()

        # Activation
        self.activation = get_activation(out_dim, activation_name)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # FiLM conditioning
        if use_film and condition_dim is not None:
            self.film = FiLMLayer(out_dim, condition_dim)
        else:
            self.film = None

    def forward(self, x, condition_emb=None):
        """
        Forward pass with optional residual connection and conditioning

        Args:
            x: Input tensor [batch_size, in_dim]
            condition_emb: Optional conditioning embedding [batch_size, condition_dim]

        Returns:
            Output tensor [batch_size, out_dim]
        """
        # Store input for potential residual connection
        residual = x if self.use_residual else None

        # Main transformation
        if self.use_gelu_gating:
            # GELU gating: gate * GELU(value)
            gate = self.linear_gate(x)
            value = self.linear_value(x)
            x = gate * F.gelu(value)
        else:
            x = self.linear(x)

        x = self.norm(x)

        # Apply activation only if not using GELU gating
        if not self.use_gelu_gating:
            x = self.activation(x)

        # Apply FiLM conditioning if enabled
        if self.film is not None and condition_emb is not None:
            x = self.film(x, condition_emb)

        x = self.dropout(x)

        # Add residual connection if enabled
        if self.use_residual and residual is not None:
            x = x + residual

        return x


class MLPAutoencoder(nn.Module):
    """
    MLP-based autoencoder with FSQ (Finite Scalar Quantization) bottleneck
    for compressing EEG patches. Includes optional FiLM conditioning based on sampling rate.
    """

    def __init__(
        self,
        patch_size: int,
        levels: list = [8, 8, 8, 5, 5, 5],  # FSQ levels
        use_conditioning: bool = False,
        condition_dim: int = 16,  # Dimension for sampling rate embedding
        max_sampling_rate: float = 1000.0,  # Maximum expected sampling rate for normalization
        encoder_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
    ):
        super().__init__()

        codebook_dim = len(levels)
        self.patch_size = patch_size
        self.codebook_dim = codebook_dim
        self.use_conditioning = use_conditioning
        self.max_sampling_rate = max_sampling_rate
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        # Sampling rate embedding if conditioning is used
        if use_conditioning:
            self.sr_embedding = nn.Sequential(
                nn.Linear(1, condition_dim),
                nn.ReLU(),
                nn.Linear(condition_dim, condition_dim),
            )

        # Encoder: compress patch_size -> codebook_dim
        encoder_blocks = []
        in_dim = patch_size

        for i, hidden_dim in enumerate(encoder_config["hidden_dims"]):
            block = ResidualMLPBlock(
                in_dim=in_dim,
                out_dim=hidden_dim,
                activation_name=encoder_config["activation"],
                dropout=encoder_config["dropout"],
                norm_type=encoder_config["norm_type"],
                use_residual=encoder_config["use_residual"],
                use_gelu_gating=encoder_config["use_gelu_gating"],
                use_film=use_conditioning,
                condition_dim=condition_dim if use_conditioning else None,
            )
            encoder_blocks.append(block)
            in_dim = hidden_dim

        # Final projection to codebook dimension
        self.encoder_proj = nn.Linear(in_dim, codebook_dim)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # Add projection layer from encoder output to FSQ dimension
        self.fsq_dim = len(levels)
        self.to_fsq = nn.Linear(codebook_dim, self.fsq_dim)
        self.from_fsq = nn.Linear(self.fsq_dim, codebook_dim)

        # FSQ quantization layer
        self.quantizer = FSQ(levels=levels)

        # Decoder: decompress codebook_dim -> patch_size
        decoder_blocks = []

        # Start with the first hidden dimension
        first_hidden_dim = decoder_config["hidden_dims"][0]
        self.decoder_proj = nn.Linear(codebook_dim, first_hidden_dim)
        in_dim = first_hidden_dim

        # Skip the first dimension since we already projected to it
        remaining_dims = list(reversed(decoder_config["hidden_dims"]))[1:]
        for i, hidden_dim in enumerate(remaining_dims):
            block = ResidualMLPBlock(
                in_dim=in_dim,
                out_dim=hidden_dim,
                activation_name=decoder_config["activation"],
                dropout=decoder_config["dropout"],
                norm_type=decoder_config["norm_type"],
                use_residual=decoder_config["use_residual"],
                use_gelu_gating=decoder_config["use_gelu_gating"],
                use_film=use_conditioning,
                condition_dim=condition_dim if use_conditioning else None,
            )
            decoder_blocks.append(block)
            in_dim = hidden_dim

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_proj = nn.Linear(in_dim, patch_size)

    def _get_sampling_rate_embedding(self, sampling_rate):
        """Convert sampling rate to embedding"""
        # Normalize sampling rate to [0, 1] range
        normalized_sr = sampling_rate / self.max_sampling_rate
        normalized_sr = torch.clamp(normalized_sr, 0, 1)

        # Add batch dimension if needed
        if normalized_sr.dim() == 0:
            normalized_sr = normalized_sr.unsqueeze(0)
        if normalized_sr.dim() == 1:
            normalized_sr = normalized_sr.unsqueeze(-1)

        return self.sr_embedding(normalized_sr)

    def encode(self, x, sampling_rate=None):
        """
        Encode input to latent space

        Args:
            x: Input tensor of shape [batch_size, patch_size]
            sampling_rate: Sampling rate tensor of shape [batch_size] or scalar
        """
        condition_emb = None
        if self.use_conditioning and sampling_rate is not None:
            condition_emb = self._get_sampling_rate_embedding(sampling_rate)

        h = x
        for block in self.encoder_blocks:
            h = block(h, condition_emb)

        return self.encoder_proj(h)

    def quantize(self, z):
        """Quantize latent codes using FSQ"""
        # Project to FSQ dimension
        z_fsq = self.to_fsq(z)

        # Add sequence dimension for FSQ (expects 3D: batch, seq, features)
        z_fsq = z_fsq.unsqueeze(1)  # [batch, 1, fsq_dim]

        quantized_fsq, indices = self.quantizer(z_fsq)

        # Remove sequence dimension and project back to codebook dimension
        quantized_fsq = quantized_fsq.squeeze(1)  # [batch, fsq_dim]
        quantized = self.from_fsq(quantized_fsq)

        # Indices also need sequence dimension removed
        indices = indices.squeeze(1)  # [batch, 1] -> [batch]

        return quantized, indices

    def decode(self, z, sampling_rate=None):
        """
        Decode from latent space

        Args:
            z: Latent codes of shape [batch_size, codebook_dim]
            sampling_rate: Sampling rate tensor of shape [batch_size] or scalar
        """
        condition_emb = None
        if self.use_conditioning and sampling_rate is not None:
            condition_emb = self._get_sampling_rate_embedding(sampling_rate)

        h = self.decoder_proj(z)
        for block in self.decoder_blocks:
            h = block(h, condition_emb)

        return self.output_proj(h)

    def forward(self, x, sampling_rate=None):
        """
        Forward pass through autoencoder

        Args:
            x: Input tensor of shape [batch_size, patch_size]
            sampling_rate: Optional sampling rate for conditioning

        Returns:
            reconstruction: Reconstructed input
            quantized: Quantized latent codes
            indices: Quantization indices
        """
        # Encode
        z = self.encode(x, sampling_rate)

        # Quantize
        quantized, indices = self.quantize(z)

        # Decode
        reconstruction = self.decode(quantized, sampling_rate)

        return reconstruction, quantized, indices

    def get_codebook_usage(self, dataloader, device="cuda"):
        """
        Compute codebook usage statistics

        Args:
            dataloader: DataLoader to compute statistics on
            device: Device to run computation on

        Returns:
            usage_stats: Dictionary with codebook usage statistics
        """
        self.eval()
        all_indices = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        x, sampling_rate = batch[0].to(device), batch[1].to(device)
                    else:
                        x, sampling_rate = batch[0].to(device), None
                else:
                    x, sampling_rate = batch.to(device), None

                _, _, indices = self.forward(x, sampling_rate)
                all_indices.append(indices.cpu())

        all_indices = torch.cat(all_indices, dim=0)

        # Calculate usage statistics
        unique_codes = torch.unique(all_indices.flatten())
        total_possible_codes = torch.prod(torch.tensor(self.quantizer.levels))
        usage_rate = len(unique_codes) / total_possible_codes.item()

        return {
            "unique_codes_used": len(unique_codes),
            "total_possible_codes": total_possible_codes.item(),
            "usage_rate": usage_rate,
            "indices_shape": all_indices.shape,
        }


if __name__ == "__main__":
    # Example usage
    patch_size = 128
    encoder_config = {
        "hidden_dims": [256, 128],
        "dropout": 0.1,
        "norm_type": "rms",
        "use_residual": True,
        "use_gelu_gating": False,
        "activation": "gelu",
    }
    decoder_config = {
        "hidden_dims": [128, 256],
        "dropout": 0.1,
        "norm_type": "rms",
        "use_residual": True,
        "use_gelu_gating": True,
        "activation": "gelu",
    }

    model = MLPAutoencoder(
        patch_size=patch_size,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        use_conditioning=True,
        condition_dim=16,
    )

    print(model)
