import os
from functools import partial

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from litdata import StreamingDataLoader, StreamingDataset
from omegaconf import DictConfig, OmegaConf

from lightning_module import EEGAutoencoderLightning


def transform(sample, patch_size):
    length = len(sample[0])
    signal = sample[0]
    if length > patch_size:
        start = torch.randint(0, length - patch_size + 1, (1,)).item()
        signal = signal[start : start + patch_size]
    return signal, sample[1], sample[2]


def create_dataloader(cfg: DictConfig, patch_size: int = 100):
    train_dataset = StreamingDataset(
        cfg.data.train_data_path,
        shuffle=True,
        drop_last=True,
        transform=partial(transform, patch_size=patch_size),
    )
    val_dataset = StreamingDataset(
        cfg.data.val_data_path,
        shuffle=False,
        drop_last=False,
        transform=partial(transform, patch_size=patch_size),
    )

    train_loader = StreamingDataLoader(
        train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    val_loader = StreamingDataLoader(
        val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )

    return train_loader, val_loader


def setup_callbacks(cfg: DictConfig, save_dir: str):
    """Setup training callbacks"""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="eeg-autoencoder-{step:06d}",
        save_last=True,
        every_n_train_steps=cfg.training.checkpoint_every_n_steps,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration
    """

    # Print configuration if requested
    if cfg.get("print_config", True):
        print("=" * 80)
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

    # Set random seeds for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Get current working directory (Hydra changes it)
    save_dir = os.getcwd()
    print(f"Working directory: {save_dir}")

    # Instantiate logger
    logger = None
    if "logger" in cfg and cfg.logger is not None:
        logger = WandbLogger(
            project=cfg.logger.project,
            name=cfg.logger.name,
            save_dir=cfg.logger.save_dir,
            log_model=cfg.logger.log_model,
            tags=cfg.logger.tags,
        )
        # Update save_dir for logger
        if hasattr(logger, "save_dir"):
            logger._save_dir = save_dir

    # Create Lightning module by directly passing the model and training config
    print("Creating Lightning module...")
    model = EEGAutoencoderLightning(
        model_config=cfg.model, log_every_n_steps=cfg.training.log_every_n_steps
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloader(cfg, patch_size=cfg.model.patch_size)

    # Setup callbacks
    callbacks = setup_callbacks(cfg, save_dir)

    # Update trainer config with logger and callbacks
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)

    # Create trainer
    print("Creating trainer...")
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **trainer_config)

    # Log hyperparameters
    if logger:
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if hasattr(logger, "experiment"):
            logger.experiment.log(
                {
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                    "model/parameters_mb": total_params * 4 / (1024**2),
                }
            )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train()
