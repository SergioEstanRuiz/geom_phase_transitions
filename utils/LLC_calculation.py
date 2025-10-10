from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce
import torch
from torch.utils.data import DataLoader
from arithmetic_models import MLP
import pandas as pd

# with torch.cpu.amp.autocast(dtype=torch.bfloat16):
def compute_and_save_llc(params):
    train_data = torch.load(f"./results/{params.exp_name}/datasets/train_data.pt")
    llcs = []
    n_checkpoints = params.n_save_model_checkpoints
    check_points_every = params.epochs // params.print_times
    model = MLP(params)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model.eval()
    for i in range(n_checkpoints):
        try:
            model.load_state_dict(torch.load(f"./results/{params.exp_name}/checkpoints/model_{(i+1) * check_points_every}.pt"))
        except FileNotFoundError:
            continue
        llcs.append(estimate_learning_coeff_with_summary(
            model=model,
            loader=DataLoader(train_data, batch_size=params.batch_size, shuffle=True),
            evaluate=evaluate_ce,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=0.003, nbeta=2.0, localization=5.0),
            num_chains=3,
            num_draws=500,
            device=DEVICE,
            online=False,
        )['llc/mean'])

    df_loss = pd.read_csv(f"./results/{params.exp_name}/loss_data.csv")
    df_loss["llc"] = llcs
    df_loss.to_csv(f"./results/{params.exp_name}/loss_data.csv", index=False)

    return df_loss