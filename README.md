## Geometric Phase Transitions in Grokking (ICML)
This repository accompanies our ICML paper on geometric phase transitions during grokking. We study modular arithmetic tasks through singular learning theory, estimate the local learning coefficient (LLC) with stochastic gradient Langevin dynamics, and track how loss landscapes evolve during training.

## Repository Structure
- `main.py`: single-run entrypoint (training + LLC logging + result saving).
- `training.py`: training loop, evaluation, and LLC sampling/plotting.
- `arithmetic_models.py`: model definitions (paper model, MLP, transformer variants).
- `utils/`: dataset generation, LLC utilities, plotting helpers.
- `sweeps/`: W&B sweep configs and `sweep_main.py` entrypoint.
- `results/`: outputs for each run or sweep (`loss_data.csv`, `params.csv`, plots).

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # torch>=2.1 recommended for devinterp
```
Authentication for Weights & Biases is required for logging:
```bash
wandb login
```

## Running a Single Experiment
```bash
source venv/bin/activate
python main.py
```
Key parameters live in `training.ExperimentParams` (e.g., `p`, `epochs`, `lr`, `model_type`, `num_chains`, `num_draws`). Results are written to `results/<exp_name>/`:
- `loss_data.csv`: training/validation metrics and LLC per checkpoint.
- `params.csv`: run hyperparameters.
- `loss_trace_epoch_*.png`: LLC trace plots.

## Sweeps
We provide a grid search over learning rate and weight decay (see `sweeps/sweep.yaml`). To launch:
```bash
source venv/bin/activate
wandb sweep sweeps/sweep.yaml          # creates a sweep id
CUDA_VISIBLE_DEVICES=<gpu_id> wandb agent <sweep_id>
```
Each sweep agent runs `sweeps/sweep_main.py`, which derives `exp_name` from the sampled config and saves results under `results/<exp_name>/`.

## Reproducing Figures
- LLC traces and metrics are generated during training and stored in `results/`.
- Notebook utilities for paper plots: `plots_for_paper.ipynb`, `visualising_results.ipynb`, and `analyse_sweep.ipynb`.
