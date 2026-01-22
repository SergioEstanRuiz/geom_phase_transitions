from utils.produce_datasets import make_dataset, train_test_split
from training import train, ExperimentParams
import torch
import os
import math
import pandas as pd
from dataclasses import asdict
from utils.LLC_calculation import compute_and_save_llc
from utils.drawing import draw
import wandb


if __name__ == "__main__":

    params = ExperimentParams() # set parameters for the experiment
    os.makedirs(name=f"./results/{params.exp_name}", exist_ok=True) # create directory for checkpoints
    torch.manual_seed(params.random_seed) # set random seed for reproducibility

    run = wandb.init(project="geom_phase_transitions", config=asdict(params))
    dataset = make_dataset(params.p)
    train_data, test_data = train_test_split(dataset, params.train_frac, params.random_seed)

    #--derived scaling law quantities (M, N, etc)--
    M = int(params.p)
    N = len(train_data)
    total_pairs = len(dataset)
    eps = 1e-12
    mlogm = M * max(math.log(M + eps), eps)

    run.config.update(
        {
            "derived/M": M, 
            "derived/N": N,
            "derived/total_pairs": total_pairs, 
            "derived/N_over_M2": N / (M * M),
            "derived/N_over_MlogM": N / mlogm,
        }
    )

    df = train(
        train_dataset=train_data, test_dataset=test_data, params=params, run=run
    )

    # Save results
    df.to_csv(f"./results/{params.exp_name}/loss_data.csv", index=False)
    df_params = pd.DataFrame([asdict(params)])  # Convert params to dictionary and then to DataFrame
    df_params.to_csv(f"./results/{params.exp_name}/params.csv", index=False)

    run.finish()

