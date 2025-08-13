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
    p: int = 53
    epochs: int = 25000
    n_save_model_checkpoints: int = 100
    print_times: int = 100
    lr: float = 0.005
    batch_size: int = 128
    hidden_size: int = 48
    embed_dim: int = 12
    train_frac: float = 0.4
    random_seed: int = 0 # Some seeds might not show grokking, or might appear later. 
    device: str = DEVICE
    weight_decay: float = 0.0002
    exp_name: str = "arithmetic_experiment_test"

    # Transformer specific parameters
    use_transformer: bool = False  # Flag to use Transformer instead of MLP
    num_heads: int = 4
    num_layers: int = 2

    