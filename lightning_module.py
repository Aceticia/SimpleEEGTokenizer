import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from autoencoder import MLPAutoencoder


class EEGAutoencoderLightning(L.LightningModule):
    """
    PyTorch Lightning module for training EEG autoencoder with FSQ quantization
    """

    def __init__(
        self,
        model_config: dict,
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 1e-6,
        log_every_n_steps: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store the model
        self.autoencoder = MLPAutoencoder(**model_config)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.log_every_n_steps = log_every_n_steps

        # Initialize comprehensive torchmetrics
        metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x, sampling_rate=None):
        """Forward pass through the autoencoder"""
        return self.autoencoder(x, sampling_rate)

    def _compute_loss(self, batch):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                x, sampling_rate = batch[0], batch[1]
            else:
                x, sampling_rate = batch[0], None
        else:
            x, sampling_rate = batch, None

        # Forward pass
        reconstruction, quantized, indices = self.forward(x, sampling_rate)

        # Compute MSE loss
        mse_loss = F.mse_loss(reconstruction, x)

        return mse_loss, reconstruction, x, sampling_rate, quantized, indices

    def _step(self, batch, batch_idx, prefix="train"):
        """Training step"""
        loss, reconstruction, original, sampling_rate, quantized, indices = (
            self._compute_loss(batch)
        )

        metrics_dict = getattr(self, f"{prefix}_metrics")
        metrics_dict.update(reconstruction, original)
        metrics = metrics_dict.compute() | {"loss": loss.item()}
        self.log_dict({f"{prefix}/{k}": v for k, v in metrics.items()})
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="val")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
