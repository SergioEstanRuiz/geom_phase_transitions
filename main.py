from utils.produce_datasets import make_dataset, train_test_split
from training import train, ExperimentParams
import torch
import os
import pandas as pd
from dataclasses import asdict
from utils.LLC_calculation import compute_and_save_llc
from utils.drawing import draw

# FOR SOME REASON YOU NEED TO DO venv/bin/python main.py
repeat = False
p_list = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]  # List of prime numbers to experiment with
for p in p_list:
    params = ExperimentParams() # set parameters for the experiment
    params.p = p
    params.exp_name = f"arithmetic_experiment_p={p}"
    os.makedirs(f"./results/{params.exp_name}/checkpoints", exist_ok=True) # create directory for checkpoints
    torch.manual_seed(params.random_seed) # set random seed for reproducibility

    # check if loss_data.csv already exists
    if os.path.exists(f"./results/{params.exp_name}/loss_data.csv") and not repeat:
        print(f"Experiment {params.exp_name} already exists. Skipping...")
    else:
        print(f"Running experiment {params.exp_name}...")
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
    try:
        df_loss = compute_and_save_llc(params)
    except Exception as e:
        print(f"Could not compute llc: {e}")
    try:
        draw(df_loss, params.exp_name)
    except Exception as e:
        print(f"Could not draw llc plot: {e}")