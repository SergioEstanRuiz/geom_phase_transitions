from utlis.produce_datasets import make_dataset, train_test_split
from training import train, ExperimentParams
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

def train(train_dataset, test_dataset, params, verbose=True):
    # all_models = []
    if params.use_transformer:
        model = Transformer(params).to(params.device)
    else:
        model = MLP(params).to(params.device)
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
            loss_data.append(
                {
                    "batch": i + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
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