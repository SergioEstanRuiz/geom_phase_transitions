# Geometric Phase Transitions

This repository studies geometric phase transitions in neural networks during training through the lens of Singular Learning Theory (SLT). We analyze various models and phenomena, including grokking in modular arithmetic and stagewise development in LLMs.

## Repository Structure

```
geom_phase_transitions/
├── configs.py              # Configuration classes and parameters
├── train.py                # Main training script for neural networks
├── arithmetic_models.py    # MLP and Transformer model definitions
├── LLC_calculation.py      # Learning Learning Coefficient calculation
├── utils/
│   └── produce_datasets.py # Dataset generation utilities
└── results/                # Training outputs and checkpoints
    ├── [experiment_name]/
    │   ├── checkpoints/     # Model checkpoints (.pt files)
    │   ├── loss_data.csv    # Training metrics over time
    │   └── params.csv       # Experiment parameters
    └── LLC/                 # LLC calculation results
        └── [model_name]/
            └── llc_results.csv
```

## Features

- **Dual Architecture Support**: Train either MLP or Transformer models on modular arithmetic
- **Automatic Device Detection**: CUDA > MPS (Apple Silicon) > CPU fallback
- **Wandb Integration**: Comprehensive experiment tracking with live URLs
- **LLC Analysis**: Bayesian model complexity analysis using SGLD sampling
- **Auto-naming**: Prevents experiment conflicts with automatic serial numbering
- **Checkpoint Management**: Configurable model saving throughout training

## Quick Start

### 1. Basic Training

```bash
# Train with default Transformer configuration
python train.py
```

This will:
- Train a Transformer on modular addition (mod 53)
- Save checkpoints every 250 epochs
- Log metrics to wandb with live dashboard URL
- Save results to `results/Transformer_Experiment_p53_XXX/`

### 2. Calculate Learning Learning Coefficients

```bash
# Calculate LLC for all trained models
python LLC_calculation.py
```

This will:
- Automatically find all trained models in `results/`
- Calculate LLC using SGLD sampling
- Create wandb visualizations comparing LLC with training metrics
- Save results to `results/LLC/[model_name]/`

## Configuration

### Model Architecture (`configs.py`)

```python
@dataclass
class ExperimentParams:
    # Model selection
    use_transformer: bool = True        # True = Transformer, False = MLP
    
    # Architecture parameters
    p: int = 53                        # Prime for modular arithmetic
    embed_dim: int = 12                # Embedding dimension
    
    # Transformer-specific (used when use_transformer=True)
    num_heads: int = 4                 # Attention heads
    num_layers: int = 2                # Transformer layers
    
    # MLP-specific (used when use_transformer=False)
    hidden_size: int = 48              # Hidden layer size
    activation: str = "relu"           # "relu", "gelu", or "quad"
    
    # Training configuration
    epochs: int = 25000                # Training epochs
    lr: float = 0.005                  # Learning rate
    batch_size: int = 128              # Batch size
    train_frac: float = 0.4            # Fraction of data for training
    
    # Logging and checkpointing
    n_save_model_checkpoints: int = 100  # Number of checkpoints to save
    print_times: int = 100               # Number of times to print metrics
    log_wandb: bool = True               # Enable wandb logging
```

### LLC Parameters

```python
@dataclass
class LLCParams:
    batch_size: int = 8               # LLC calculation batch size
    llc_lr: float = 0.003             # SGLD learning rate
    llc_nbeta: float = 2.0            # Temperature parameter
    llc_localization: float = 5.0     # Localization strength
    num_chains: int = 2               # Number of MCMC chains
    num_draws: int = 100              # Samples per chain
    log: bool = True                  # Enable wandb logging
```

## Custom Experiments

### Example: MLP with Different Prime

```python
from configs import ExperimentParams
from train import train
from utils.produce_datasets import make_dataset, train_test_split

# Custom MLP configuration
params = ExperimentParams(
    use_transformer=False,    # Use MLP instead
    p=97,                     # Different prime
    hidden_size=64,           # Larger hidden layer
    activation="gelu",        # Different activation
    epochs=50000,             # Longer training
    exp_name="custom_mlp_p97" # Custom name
)

# Generate dataset and train
full_dataset = make_dataset(params.p)
train_dataset, test_dataset = train_test_split(full_dataset, params.train_frac)
results = train(train_dataset, test_dataset, params)
```

### Example: Large Transformer

```python
params = ExperimentParams(
    use_transformer=True,
    p=113,                    # Larger prime
    embed_dim=24,            # Larger embeddings
    num_heads=8,             # More attention heads
    num_layers=4,            # Deeper model
    batch_size=256,          # Larger batches
    lr=0.001                 # Lower learning rate
)
```

## Wandb Integration

The repository includes comprehensive wandb integration:

### Training Visualization
- Live training/validation loss and accuracy
- Model architecture summaries
- Hyperparameter tracking
- Real-time dashboard URLs printed to terminal

### LLC Analysis Visualization
- LLC values over training checkpoints
- Combined plots of LLC vs training metrics
- Phase transition identification
- Model complexity evolution

### Setup Wandb

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Update `wandb_entity` in `configs.py` with your username

## Understanding the Output

### Training Results
Each experiment creates:
```
results/Transformer_Experiment_p53_001/
├── checkpoints/
│   ├── checkpoint_250.pt   # Model at epoch 250
│   ├── checkpoint_500.pt   # Model at epoch 500
│   └── ...
├── loss_data.csv          # Training metrics every 250 epochs
└── params.csv            # Complete experiment configuration
```

### LLC Results
```
results/LLC/Transformer_Experiment_p53_001/
└── llc_results.csv       # LLC values for each checkpoint
```

### Training Summary Example
```
TRAINING SETUP
============================================================
Device: CUDA (NVIDIA GPU)

Model Architecture:
   Type: Transformer
   Parameters: 8,429
   Embedding Dim: 12
   Heads: 4
   Layers: 2

Training Configuration:
   Epochs: 25,000
   Learning Rate: 0.005
   Batch Size: 128
   Weight Decay: 0.0002

Dataset:
   Modular Arithmetic (p=53)
   Training samples: 1,113
   Test samples: 1,696
============================================================

🚀 Wandb logging enabled!
📊 View results at: https://wandb.ai/your-username/geom_phase_transitions/runs/xyz
```

## Performance Tips

### For Training
- Use CUDA if available for faster training
- Increase `batch_size` if you have sufficient GPU memory
- Adjust `print_times` to control logging frequency

### For LLC Calculation
- LLC computation is memory-intensive; reduce `batch_size` if needed
- Increase `num_chains` and `num_draws` for more accurate estimates
- Use smaller models for faster LLC computation

## Dependencies

```bash
pip install torch transformers wandb pandas tqdm devinterp
```

## Research Applications

This codebase is designed for studying:
- **Grokking phenomena** in neural networks
- **Phase transitions** during training
- **Architecture comparisons** between MLPs and Transformers
- **Modular arithmetic** as a controlled learning task

## To Do

- [ ] Change hyperparameter system to use `config.yaml` files instead of Python dataclasses
- [ ] Refactor LLC functionality into separate modules
- [ ] Create visualization-only module for analyzing saved calculation results
- [ ] Define project roadmap and create contribution tickets
- [ ] Add support for additional model architectures
- [ ] Implement automated hyperparameter sweeps

## Contributing

The repository uses automatic experiment naming to prevent conflicts. Each run generates a unique experiment name with serial numbers (e.g., `Transformer_Experiment_p53_001`), making it safe to run multiple experiments without overwriting results.

For contributing to this research project, please see the To Do section above for current priorities and planned features. We welcome contributions that align with the project's aims, especially in expanding model support, enhancing LLC analysis, and improving configuration flexibility.
