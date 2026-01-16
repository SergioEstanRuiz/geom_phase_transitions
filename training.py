from copy import deepcopy
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from arithmetic_models import paperModel, centred_loss, MLP, transformerModel
from dataclasses import dataclass
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import default_nbeta
import warnings
import os
from utils.metrics import grokking_test3 as grokking_test


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The @dataclass decorator automatically generates special methods for the class, such as __init__ and __repr__.
# This is useful for classes that are primarily used to store data without much additional functionality.
@dataclass
class ExperimentParams:
    p: int = 71
    epochs: int = 25000
    checkpoint_epochs: int = 100
    lr: float = 0.005
    batch_size: int = 32
    hidden_size: int = 48
    embed_dim: int = 24
    train_frac: float = 0.4
    random_seed: int = 0 # Some seeds might not show grokking, or might appear later. 
    device: str = DEVICE
    weight_decay: float = 0.0005
    exp_name: str = "MLP_model"
    optimiser: str = "adam" # Options: 'adam', 'sgd'
    loss: str = "mse" # Options: 'cross_entropy', 'mse', but mse is centred_loss
    num_chains: int = 3
    num_draws: int = 500
    num_burnin: int = 100
    activation: str = "quadratic"  # Options: 'relu', 'quadratic'
    model_type: str = "MLP"  # Options: 'MLP', 'transformer', 'paper'
    num_layers: int = 1  # For transformer model
    nhead: int = 2       # For transformer model
    dim_feedforward: int = 128  # For transformer model

def train(train_dataset, test_dataset, params, run=None):
    warnings.filterwarnings("ignore")
    device = torch.device(params.device)
    if params.model_type == "paper":
        model = paperModel(params).to(device)
    elif params.model_type == "transformer":
        model = transformerModel(params).to(device)
    elif params.model_type == "MLP":
        model = MLP(params).to(device)

    if params.optimiser == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=params.weight_decay, lr=params.lr
        )
    elif params.optimiser == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), weight_decay=params.weight_decay, lr=params.lr
        )
    
    if params.loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif params.loss == "mse":
        loss_fn = centred_loss()

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    checkpoint_every = params.checkpoint_epochs

    llc_device = device
    amp_guard_factory = nullcontext
    if llc_device.type == "cpu":
        amp_guard_factory = _cpu_autocast_patch

    # you need model, dataset arguments for llc_calculation function
    def test(model, dataset):
        n_correct = 0
        total_loss = 0
        model.eval()

        with torch.no_grad():
            for x, y in dataset:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                total_loss += loss.item()
                pred = torch.argmax(out, dim=-1) # get the index of the maximum log-probability
                y_cat = torch.argmax(y, dim=-1)
                if pred == y_cat:
                    n_correct += 1
            return n_correct / len(dataset), total_loss / len(dataset)

    def test_for_llc(model, dataset):
        # for the llc you only evaluate one instance of x and y 
        model.eval()
        x, y = dataset
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        return loss, {"output": out} # you need this format for llc function

    loss_data = []
    recent_val_acc = []  # track last few validation accuracies for early stopping
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

        if (i + 1) % checkpoint_every == 0:
            val_acc, val_loss = test(model, test_dataset)
            train_acc, train_loss = test(model, train_dataset)
            # Avoid distributed all_reduce inside devinterp LLC callback when running single-process on GPU
            if llc_device.type == "cuda":
                os.environ.setdefault("USE_SPMD", "1")
            with amp_guard_factory():
                llc = estimate_learning_coeff_with_summary(
                        model,
                        loader=train_loader,
                        evaluate=test_for_llc,
                        sampling_method=SGLD,
                        optimizer_kwargs=dict(lr=4e-4, localization=100.0, nbeta=default_nbeta(train_loader)),
                        num_chains=params.num_chains,  # How many independent chains to run
                        num_draws=params.num_draws,  # How many samples to draw per chain
                        num_burnin_steps=params.num_burnin,  # How many samples to discard at the beginning of each chain
                        num_steps_bw_draws=1,  # How many steps to take between each sample
                        device=llc_device,
                        online=False,
                        verbose=False,
                    )['llc/mean']
            loss_data.append(
                {
                    "batch": i + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "llc": llc,
                }
            )
            run.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "llc": llc,
                }
            )
            recent_val_acc.append(val_acc)
            if len(recent_val_acc) > 3:
                recent_val_acc.pop(0)
            if len(recent_val_acc) == 3 and all(acc >= 0.95 for acc in recent_val_acc):
                print(f"Early stopping at epoch {i+1}: recent val accs {recent_val_acc}")
                break
    df = pd.DataFrame(loss_data)
    grokking = grokking_test(df["train_acc"], df["val_acc"])
    run.log({"grokking_test": grokking})
    return df

from contextlib import contextmanager, nullcontext
@contextmanager
def _cpu_autocast_patch():
    original = torch.autocast
    def patched(device_type, *args, **kwargs):
        if device_type == "cpu":
            dtype = kwargs.get("dtype")
            if len(args) > 0 and dtype is None:
                args = list(args)
                dtype = args.pop(0)
            if dtype is None or dtype is torch.float16:
                kwargs["dtype"] = torch.bfloat16
        return original(device_type, *args, **kwargs)
    torch.autocast = patched
    try:
        yield
    finally:
        torch.autocast = original
