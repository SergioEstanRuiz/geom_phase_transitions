from utils.produce_datasets import make_dataset, train_test_split
import torch
import os
import pandas as pd
from dataclasses import asdict
from copy import deepcopy
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from arithmetic_models import MLP, Transformer
from configs import get_device, ExperimentParams
import wandb

def test(model, dataset, device):
    n_correct = 0
    total_loss = 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            pred = torch.argmax(out) # get the index of the maximum log-probability
            if pred == y:
                n_correct += 1
    return n_correct / len(dataset), total_loss / len(dataset)

def print_training_summary(params, model, train_size, test_size):
    """Print summary before training starts"""
    print("=" * 60)
    print("TRAINING SETUP")
    print("=" * 60)
    
    # Device info
    device_name = str(params.device).upper()
    if "cuda" in device_name.lower():
        print(f"Device: {device_name} (NVIDIA GPU)")
    elif "mps" in device_name.lower():
        print(f"Device: {device_name} (Apple Silicon GPU)")
    else:
        print(f"Device: {device_name}")
    
    # Model info
    model_type = "Transformer" if params.use_transformer else "MLP"
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Architecture:")
    print(f"   Type: {model_type}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Embedding Dim: {params.embed_dim}")
    print(f"   Hidden Size: {params.hidden_size}")
    print(f"   Activation: {params.activation}")
    
    if params.use_transformer:
        print(f"   Heads: {params.num_heads}")
        print(f"   Layers: {params.num_layers}")
    
    print(f"\nTraining Configuration:")
    print(f"   Epochs: {params.epochs:,}")
    print(f"   Learning Rate: {params.lr}")
    print(f"   Batch Size: {params.batch_size}")
    print(f"   Weight Decay: {params.weight_decay}")
    
    print(f"\nDataset:")
    print(f"   Modular Arithmetic (p={params.p})")
    print(f"   Training samples: {train_size:,}")
    print(f"   Test samples: {test_size:,}")
    
    print("=" * 60)
    print()

def train(train_dataset, test_dataset, params, verbose=True):
    # Initialize wandb if enabled
    wandb_run = None
    if params.log_wandb:
        wandb_run = wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            name=params.exp_name,
            config=asdict(params),
            tags=["training", "arithmetic", params.activation],
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
                console="off"
            )
        )
        
        # Print wandb URL
        if wandb_run:
            print(f"Wandb logging enabled!")
            print(f"View results at: {wandb_run.url}")
            print()
    
    # all_models = []
    if params.use_transformer:
        model = Transformer(params).to(params.device)
    else:
        model = MLP(params).to(params.device)
    
    if verbose:
        print_training_summary(params, model, len(train_dataset), len(test_dataset))
    
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=params.weight_decay, lr=params.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    print_every = params.epochs // params.print_times
    checkpoint_every = None
    if params.n_save_model_checkpoints > 0:
        checkpoint_every = params.epochs // params.n_save_model_checkpoints

    loss_data = []
    if verbose:
        pbar = tqdm(total=params.epochs, desc="Training")
    for i in range(params.epochs):
        # Sample random batch of data
        batch = next(iter(train_loader))
        X, Y = batch
        X, Y = X.to(params.device), Y.to(params.device)
        # Gradient update
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            # all_models.append([deepcopy(model)])
            torch.save(model.state_dict(), f"./results/{params.exp_name}/checkpoints/model_{i + 1}.pt")
            if verbose:
                pbar.write(f"Checkpoint saved at epoch {i + 1}")
            else:
                print(f"Checkpoint saved at epoch {i + 1}")

        if (i + 1) % print_every == 0:
            val_acc, val_loss = test(model, test_dataset, params.device)
            train_acc, train_loss = test(model, train_dataset, params.device)
            
            loss_data.append({
                "epoch": i + 1, 
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
            # Log to wandb - USE SEQUENTIAL STEP INSTEAD OF EPOCH
            if params.log_wandb:
                step_index = len(loss_data)  # This will be 1, 2, 3, ..., 100
                wandb.log({
                    "epoch": i + 1,          # Still log the actual epoch
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, step=step_index)         # Use sequential step instead of i+1
            
            if verbose:
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_loss:.4f}",
                        "train_acc": f"{train_acc:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.4f}",
                    }
                )
                pbar.update(print_every)
    if verbose:
        pbar.close()
    df = pd.DataFrame(loss_data)
    train_acc, train_loss = test(model, train_dataset, params.device)
    val_acc, val_loss = test(model, test_dataset, params.device)
    if verbose:
        print(f"Final Train Acc: {train_acc:.4f} | Final Train Loss: {train_loss:.4f}")
        print(f"Final Val Acc: {val_acc:.4f} | Final Val Loss: {val_loss:.4f}")
    
    # Print final wandb link
    if params.log_wandb and wandb_run:
        print()
        print("=" * 60)
        print("✅ Training completed!")
        print(f"📊 View detailed results at: {wandb_run.url}")
        print("=" * 60)
    
    if params.log_wandb:
        wandb.finish()
    
    return df

DEVICE = get_device()  # Get the best available device

params = ExperimentParams() # set parameters for the experiment
os.makedirs(f"./results/{params.exp_name}/checkpoints", exist_ok=True) # create directory for checkpoints
torch.manual_seed(params.random_seed) # set random seed for reproducibility

dataset = make_dataset(params.p)
train_data, test_data = train_test_split(dataset, params.train_frac, params.random_seed)
# Save the dataset
os.makedirs(f"./results/{params.exp_name}/datasets", exist_ok=True)
torch.save(train_data, f"./results/{params.exp_name}/datasets/train_data.pt")
torch.save(test_data, f"./results/{params.exp_name}/datasets/test_data.pt")

df = train(
    train_dataset=train_data, test_dataset=test_data, params=params
)

# Save results
df.to_csv(f"./results/{params.exp_name}/loss_data.csv", index=False)
df_params = pd.DataFrame([asdict(params)])  # Convert params to dictionary and then to DataFrame
df_params.to_csv(f"./results/{params.exp_name}/params.csv", index=False)