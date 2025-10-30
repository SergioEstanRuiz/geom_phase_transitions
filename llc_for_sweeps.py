import os 
import pandas as pd
from utils.LLC_calculation import compute_and_save_llc
from training import ExperimentParams
import torch

path = "/home/se24/geom_phase_transitions/results/sweep2/"
for folder in os.listdir(path):
    full_path = os.path.join(path, folder)
    df_params = pd.read_csv(os.path.join(full_path, "params.csv"))
    params_dict = df_params.to_dict(orient="list")
    for key in params_dict:
        params_dict[key] = params_dict[key][0]
    params = ExperimentParams(**params_dict)
    print(f"Computing LLC for sweep in folder: {folder}")
    df_loss = compute_and_save_llc(params)
