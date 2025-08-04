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
        log_every_n_steps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = lr
        self.weight_decay = weight_decay

        # Store the model
        self.autoencoder = MLPAutoencoder(**model_config)
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

    def forward(self, x, sampling_rate=None, data_type=None):
        """Forward pass through the autoencoder"""
        return self.autoencoder(x, sampling_rate, data_type)

    def _compute_loss(self, batch):
        x, sfreq, data_type = batch
        reconstruction, quantized, indices = self.forward(x, sfreq, data_type)

        # Compute MSE loss
        mse_loss = F.mse_loss(reconstruction, x.to(reconstruction.dtype))

        return mse_loss, reconstruction, x, sfreq, quantized, indices

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
