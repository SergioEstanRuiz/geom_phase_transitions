from dataclasses import dataclass
import torch


def get_device():     # Maybe move to a utils.py file later.
    if torch.cuda.is_available():
        return torch.device("cuda")    
    elif torch.backends.mps.is_available():
        try:
            # Test if MPS is actually functional
            test_tensor = torch.tensor([1.0], device="mps")
            return torch.device("mps")
        except Exception as e:
            print(f"MPS available but not functional: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")    
    else:
        return torch.device("cpu")

DEVICE = get_device()
    
# The @dataclass decorator automatically generates special methods for the class, such as __init__ and __repr__.
# This is useful for classes that are primarily used to store data without much additional functionality.
@dataclass
class ExperimentParams:

    exp_name: str = "arithmetic_experiment_test"

    # Model parameters
    p: int = 53
    embed_dim: int = 12
    hidden_size: int = 48

    # Transformer specific parameters
    use_transformer: bool = False  # Flag to use Transformer instead of MLP
    num_heads: int = 4
    num_layers: int = 2

    # Training parameters
    epochs: int = 25000
    lr: float = 0.005
    batch_size: int = 128
    weight_decay: float = 0.0002
    train_frac: float = 0.4

    # Checkpointing and logging
    n_save_model_checkpoints: int = 100
    print_times: int = 100
    activation: str = "quad" # Options: "relu", "gelu", "quad"

    # Base parameters
    random_seed: int = 0 
    device: torch.device = DEVICE
 


@dataclass
class LLCParams:
    model_name: str = "EleutherAI/pythia-70m"
    batch_size: int = 8
    llc_lr: float = 0.003
    llc_nbeta: float = 2.0
    llc_localization: float = 5.0
    num_chains: int = 2
    num_draws: int = 100
    device: torch.device = DEVICE
    log: bool = True