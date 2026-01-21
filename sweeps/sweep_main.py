import sys
# append the project root directory to the sys.path
sys.path.append("/home/se24/geom_phase_transitions")
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
from utils.cudaStuff import get_free_gpu

# Select the computation device (ie. GPU if available)
#gpu_id = get_free_gpu() # Get the GPU with the most free memory
#device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")


def recursive_update(params: ExperimentParams, updates: dict):
    for k, v in updates.items():
        if k != "exp_name":
            setattr(params, k, v) # sets params.k = v
            params.exp_name = f"{params.exp_name}_{k}={v}" # update exp_name to reflect the change
    params.exp_name = f"{updates['exp_name']}/{params.exp_name}"

params = ExperimentParams() # set parameters for the experiment

# --- Initialise wandb with the full config ---
run = wandb.init(project="geom_phase_transitions")
config = copy.deepcopy(dict(run.config))
recursive_update(params, config)
wandb.config.update(asdict(params)) # upload the full config to wandb

os.makedirs(f"./results/{params.exp_name}", exist_ok=True) # create directory for checkpoints
torch.manual_seed(params.random_seed) # set random seed for reproducibility

dataset = make_dataset(params.p)
train_data, test_data = train_test_split(dataset, params.train_frac, params.random_seed)

df = train(
    train_dataset=train_data, test_dataset=test_data, params=params, run=run
)

# Save results
df.to_csv(f"./results/{params.exp_name}/loss_data.csv", index=False)
df_params = pd.DataFrame([asdict(params)])  # Convert params to dictionary and then to DataFrame
df_params.to_csv(f"./results/{params.exp_name}/params.csv", index=False)

run.finish()
