from utils.produce_datasets import make_dataset, train_test_split
from training import train, ExperimentParams
import torch
import os
import pandas as pd
from dataclasses import asdict
from utils.LLC_calculation import compute_and_save_llc
from utils.drawing import draw
import wandb

# FOR SOME REASON YOU NEED TO DO venv/bin/python main.py
repeat = False
p_list = [71]  # List of prime numbers to experiment with
for p in p_list:
    params = ExperimentParams() # set parameters for the experiment
    params.p = p
    params.exp_name = f"arithmetic_experiment_p={p}_sdg_mse"
    os.makedirs(f"./results/{params.exp_name}", exist_ok=True) # create directory for checkpoints
    torch.manual_seed(params.random_seed) # set random seed for reproducibility

    run = wandb.init(project="geom_phase_transitions", config=asdict(params))
    # check if loss_data.csv already exists
    if os.path.exists(f"./results/{params.exp_name}/loss_data.csv") and not repeat:
        print(f"Experiment {params.exp_name} already exists. Skipping...")
    else:
        print(f"Running experiment {params.exp_name}...")
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