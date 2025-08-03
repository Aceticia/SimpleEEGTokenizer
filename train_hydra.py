import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from lightning_module import EEGAutoencoderLightning


def create_dummy_dataloader(cfg: DictConfig):
    """
    Create a dummy dataloader for testing purposes.
    Replace this with your actual EEG data loading logic.
    """

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(
            self, num_samples: int, patch_size: int, use_conditioning: bool = False
        ):
            self.num_samples = num_samples
            self.patch_size = patch_size
            self.use_conditioning = use_conditioning

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random EEG-like data (normalized to data_range)
            x = torch.randn(self.patch_size) * 0.1
            x = torch.clamp(x, cfg.data.data_range[0], cfg.data.data_range[1])

            if self.use_conditioning:
                # Random sampling rate between 100-1000 Hz
                sampling_rate = torch.rand(1) * 900 + 100
                return x, sampling_rate
            else:
                return x

    def create_loader(num_batches: int):
        dataset = DummyDataset(
            num_samples=num_batches * cfg.data.batch_size,
            patch_size=cfg.data.patch_size,
            use_conditioning=cfg.data.use_conditioning,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.persistent_workers,
            pin_memory=cfg.data.pin_memory,
        )

    train_loader = create_loader(cfg.data.train_num_batches)
    val_loader = create_loader(cfg.data.val_num_batches)

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
        model_config=cfg.model,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        warmup_start_lr=cfg.training.warmup_start_lr,
        eta_min=cfg.training.eta_min,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dummy_dataloader(cfg)

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
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Finish wandb run if using wandb
        if (
            logger
            and hasattr(logger, "experiment")
            and hasattr(logger.experiment, "finish")
        ):
            try:
                logger.experiment.finish()
            except:
                pass

    print(f"Training artifacts saved in: {save_dir}/checkpoints/")


if __name__ == "__main__":
    train()
