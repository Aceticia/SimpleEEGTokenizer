import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.layers import Mlp
from vector_quantize_pytorch import FSQ


def modulate(x, shift, scale):
    return x * scale + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Block(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, condition_dim=None, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
            norm_layer=nn.LayerNorm,
            drop=0.0,
        )

        if condition_dim is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(condition_dim, 3 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        if self.adaLN_modulation is not None:
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
            x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        else:
            x = x + self.mlp(self.norm(x))
        x = self.dropout(x)
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
        sr_condition_dim: int = 16,
        sr_freq_dim: int = 32,
        log_sr_embedding: bool = False,
        max_sampling_rate: float = 2000.0,
        encoder_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
    ):
        super().__init__()

        codebook_dim = len(levels)
        self.patch_size = patch_size
        self.codebook_dim = codebook_dim
        self.use_conditioning = use_conditioning
        self.log_sr_embedding = log_sr_embedding
        self.sr_freq_dim = sr_freq_dim
        self.max_sampling_rate = max_sampling_rate
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        # Sampling rate embedding if conditioning is used
        if use_conditioning:
            self.sr_embedding = TimestepEmbedder(
                hidden_size=sr_condition_dim,
                frequency_embedding_size=sr_freq_dim,
            )
            self.dataset_type_embedding = nn.Embedding(
                num_embeddings=3,  # EEG, iEEG, MEG
                embedding_dim=sr_condition_dim,
            )

        # Encoder: compress patch_size -> codebook_dim
        encoder_blocks = []
        self.encoder_proj = nn.Linear(encoder_config["hidden_dim"], codebook_dim)
        self.input_proj = nn.Linear(patch_size, encoder_config["hidden_dim"])
        for _ in range(encoder_config["n_layers"]):
            block = Block(
                hidden_size=encoder_config["hidden_dim"],
                dropout=encoder_config["dropout"],
                mlp_ratio=encoder_config["mlp_ratio"],
                condition_dim=sr_condition_dim if use_conditioning else None,
            )
            encoder_blocks.append(block)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # FSQ quantization layer
        self.quantizer = FSQ(levels=levels)

        # Decoder: decompress codebook_dim -> patch_size
        decoder_blocks = []
        self.decoder_proj = nn.Linear(codebook_dim, decoder_config["hidden_dim"])
        self.output_proj = nn.Linear(decoder_config["hidden_dim"], patch_size)
        for _ in range(decoder_config["n_layers"]):
            block = Block(
                hidden_size=decoder_config["hidden_dim"],
                dropout=decoder_config["dropout"],
                mlp_ratio=decoder_config["mlp_ratio"],
                condition_dim=sr_condition_dim if use_conditioning else None,
            )
            decoder_blocks.append(block)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def _get_sampling_rate_embedding(self, sampling_rate):
        if self.log_sr_embedding:
            log_sr = torch.log1p(sampling_rate)
            normalized_sr = log_sr / math.log(self.max_sampling_rate + 1.0)
        else:
            normalized_sr = sampling_rate / self.max_sampling_rate
        out = self.sr_embedding(normalized_sr)
        return out

    def encode(self, x, sampling_rate=None, dataset_type=None):
        condition_emb = None
        if self.use_conditioning and sampling_rate is not None:
            condition_emb = self._get_sampling_rate_embedding(sampling_rate)
            condition_emb = self.dataset_type_embedding(dataset_type) + condition_emb

        h = self.input_proj(x.to(self.encoder_proj.weight.dtype))
        for block in self.encoder_blocks:
            h = block(h, condition_emb)

        return self.encoder_proj(h), condition_emb

    def quantize(self, z):
        z_fsq = z.unsqueeze(1)  # [batch, 1, fsq_dim]
        quantized_fsq, indices = self.quantizer(z_fsq)
        quantized_fsq = quantized_fsq.squeeze(1)  # [batch, fsq_dim]
        indices = indices.squeeze(1)  # [batch, 1] -> [batch]
        return quantized_fsq, indices

    def decode(self, z, sampling_rate=None, dataset_type=None, condition_emb=None):
        if condition_emb is None and self.use_conditioning:
            condition_emb = self._get_sampling_rate_embedding(sampling_rate)
            condition_emb = self.dataset_type_embedding(dataset_type) + condition_emb

        h = self.decoder_proj(z)
        for block in self.decoder_blocks:
            h = block(h, condition_emb)

        return self.output_proj(h)

    def forward(self, x, sampling_rate=None, dataset_type=None):
        z, condition_emb = self.encode(x, sampling_rate, dataset_type)
        quantized, indices = self.quantize(z)
        reconstruction = self.decode(
            quantized, sampling_rate, dataset_type, condition_emb=condition_emb
        )
        return reconstruction, quantized, indices


if __name__ == "__main__":
    # Example usage
    model = MLPAutoencoder(
        patch_size=128,
        levels=[8, 8, 8, 5, 5, 5],
        use_conditioning=True,
        sr_condition_dim=16,
        sr_freq_dim=32,
        log_sr_embedding=True,
        max_sampling_rate=2000.0,
        encoder_config={
            "hidden_dim": 256,
            "n_layers": 4,
            "dropout": 0.1,
            "mlp_ratio": 4.0,
        },
        decoder_config={
            "hidden_dim": 256,
            "n_layers": 4,
            "dropout": 0.1,
            "mlp_ratio": 4.0,
        },
    )

    # Dummy input
    x = torch.randn(10, 128)  # Batch of 10 samples,
    sampling_rate = torch.tensor([1000.0] * 10)  # Example sampling rate
    dataset_type = torch.tensor([0] * 10)  # Example dataset type
    print([a.shape for a in model(x, sampling_rate, dataset_type)])
