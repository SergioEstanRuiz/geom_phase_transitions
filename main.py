from utils.produce_datasets import make_dataset, train_test_split
from training import train, ExperimentParams
import torch
import os
import pandas as pd
from dataclasses import asdict

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