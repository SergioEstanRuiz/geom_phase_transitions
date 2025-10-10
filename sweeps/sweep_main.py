from utils.produce_datasets import make_dataset, train_test_split
from training import train, ExperimentParams
import torch
import os
import pandas as pd
from dataclasses import asdict
from utils.LLC_calculation import compute_and_save_llc
from utils.drawing import draw
import wandb
import copy

def recursive_update(params: ExperimentParams, updates: dict):
    for k, v in updates.items():
        setattr(params, k, v) # sets params.k = v

params = ExperimentParams() # set parameters for the experiment

# --- Initialise wandb with the full config ---
run = wandb.init(project="geom_phase_transitions")
config = copy.deepcopy(dict(wandb.config))
recursive_update(params, config)
wandb.config.update(asdict(params))# upload the full config to wandb

# Optional: rename run for tracking
params.exp_name = f"sweep/p={params.p}_trainfacc={params.train_frac}_wdecay={params.weight_decay}"

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
last_test_acc = df["val_acc"].values[-1]
run.log({"test_acc": last_test_acc})

# Save results
df.to_csv(f"./results/{params.exp_name}/loss_data.csv", index=False)
df_params = pd.DataFrame([asdict(params)])  # Convert params to dictionary and then to DataFrame
df_params.to_csv(f"./results/{params.exp_name}/params.csv", index=False)
try:
    df_loss = compute_and_save_llc(params)
except Exception as e:
    print(f"Could not compute llc: {e}")
try:
    draw(df_loss, params.exp_name)
except Exception as e:
    print(f"Could not draw llc plot: {e}")
if run is not None:
    run.finish()