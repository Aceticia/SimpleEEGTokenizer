# Simple EEG Tokenizer

A PyTorch Lightning implementation of an MLP-based autoencoder with FSQ (Finite Scalar Quantization) for compressing EEG patches. Features optional FiLM-style conditioning based on sampling rate, comprehensive metrics logging with wandb, and configurable training via Hydra.

## Features

- **MLP Autoencoder**: Simple yet effective architecture for EEG compression
- **FSQ Quantization**: Finite Scalar Quantization for discrete latent representations
- **FiLM Conditioning**: Optional conditioning on sampling rate using Feature-wise Linear Modulation
- **Snake Activation**: Optional learnable Snake activation function as alternative to ReLU
- **Comprehensive Metrics**: MSE, MAE, PSNR, SSIM, SNR, correlation, and codebook usage tracking
- **Hydra Configuration**: Flexible configuration management with experiment presets
- **W&B Integration**: Complete experiment tracking and visualization

## Installation

```bash
pip install -r requierments.txt
```

## Project Structure

```
SimpleEEGTokenizer/
├── autoencoder.py          # MLP autoencoder with FSQ and FiLM conditioning
├── lightning_module.py     # PyTorch Lightning training module
├── train.py               # Basic training script (legacy)
├── train_hydra.py         # Hydra-based training script
├── config/                # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── model/             # Model configurations
│   ├── optimizer/         # Optimizer configurations  
│   ├── scheduler/         # Scheduler configurations
│   ├── trainer/           # Trainer configurations
│   ├── data/              # Data configurations
│   ├── logger/            # Logger configurations
│   └── experiment/        # Experiment presets
└── requierments.txt       # Python dependencies
```

## Quick Start

### Basic Training

```bash
# Train with default configuration
python train_hydra.py

# Train with specific experiment
python train_hydra.py --config-name config experiment=snake_activation

# Train with conditioning
python train_hydra.py --config-name config experiment=conditioning

# Override specific parameters
python train_hydra.py model.learning_rate=0.0005 model.activation=snake

# Disable wandb logging
python train_hydra.py logger=null
```

### Configuration Examples

1. **Default Training**:
   ```bash
   python train_hydra.py
   ```

2. **Snake Activation Experiment**:
   ```bash
   python train_hydra.py experiment=snake_activation
   ```

3. **Sampling Rate Conditioning**:
   ```bash
   python train_hydra.py experiment=conditioning
   ```

4. **Custom Configuration**:
   ```bash
   python train_hydra.py \
     model.patch_size=512 \
     model.hidden_dims=[1024,512,256] \
     model.learning_rate=0.001 \
     data.batch_size=64 \
     trainer.max_epochs=200
   ```

## Model Architecture

### MLP Autoencoder
- **Encoder**: Configurable MLP layers with dropout
- **Quantizer**: FSQ with configurable levels 
- **Decoder**: Symmetric MLP architecture
- **Activations**: ReLU or learnable Snake activation

### FiLM Conditioning (Optional)
When enabled, sampling rate is embedded and used to modulate features:
```python
# Enable conditioning
model.use_conditioning=true
```

### FSQ Quantization
Finite Scalar Quantization with configurable levels:
```yaml
model:
  levels: [8, 8, 8, 5, 5, 5]  # 6D codebook with specified levels per dimension
```

## Configuration

The project uses Hydra for configuration management. Key configuration groups:

### Model Configuration (`config/model/`)
```yaml
model:
  patch_size: 256
  hidden_dims: [512, 256, 128] 
  levels: [8, 8, 8, 5, 5, 5]
  activation: "relu"  # or "snake"
  use_conditioning: false
```

### Training Configuration (`config/trainer/`)
```yaml
trainer:
  max_epochs: 100
  accelerator: "auto"
  precision: "16-mixed"
  gradient_clip_val: 1.0
```

## Metrics

The system tracks comprehensive reconstruction metrics:

- **Loss Metrics**: MSE, MAE, RMSE, NRMSE
- **Training Metrics**: Learning rate, gradient norms

## Data Loading

Currently includes dummy data loader for testing. Replace `create_dummy_dataloader()` in `train_hydra.py` with your EEG data loading logic:

```python
def create_dummy_dataloader(cfg: DictConfig):
    # Replace with your EEG data loading
    # Should return train_loader, val_loader
    pass
```

Expected data format:
- **Input**: `[batch_size, patch_size]` tensor
- **Conditioning** (optional): `[batch_size, 1]` sampling rate tensor

## Advanced Usage

### Hyperparameter Sweeps
```bash
# Create sweep configuration
python train_hydra.py --multirun \
  model.learning_rate=0.001,0.0005,0.0001 \
  model.activation=relu,snake
```

### Resume Training
```bash
python train_hydra.py trainer.resume_from_checkpoint=path/to/checkpoint.ckpt
```

### Custom Scheduler
```yaml
# config/scheduler/cosine.yaml
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: ${model.max_epochs}
  eta_min: ${model.eta_min}
```

## Development

### Adding New Activation Functions
1. Implement in `autoencoder.py`
2. Update `get_activation()` function
3. Add to configuration choices

### Adding New Metrics
1. Import in `lightning_module.py`
2. Add to metric collections
3. Update logging in training/validation steps

## Citation

If you use this code in your research, please cite:
```
@software{simple_eeg_tokenizer,
  title={Simple EEG Tokenizer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourname/SimpleEEGTokenizer}
}