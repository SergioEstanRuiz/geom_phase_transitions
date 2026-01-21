### Quick intro 
The aim of this repo is to study geometric phase transitions in models during training, through the lens of singular learning theory. To this end, we are focusing on the phenomena of grokking as observed in a modular arithmetic task.

<<<<<<< HEAD
### How this repo is structured: 
Honestly, it's still a bit of a mess. We'll get it sorted it! Some information:
- The idea is for `main.py` to control everything. Right now it only trains models. We could have it so that it can train models and then calculates LLC curves. This file calls auxiliary scripts like `training.py`, which controls how the training of the model is done, or `arithmetic_models.py`, where the different models used in the arithmetic task are placed, or `produce_datasets.py` which creates the modular arithmetic datasets. 
- On the other hand, we have the `visualising_results.ipynb` which right now does the LLC calculations and then plots curves. 
=======

### How to do sweeps:
- `source venv/bin/activate; wandb sweep ./sweeps/sweep.yaml` --> creates the sweep
- `screen -S agent1`--> creates a screen named agent1
- `nvidia-smi` --> check which GPU has the most available memory
- `source venv/bin/activate` --> activate the virtual environment if you're using one
- `CUDA_VISIBLE_DEVICES=2 wandb agent 's-estan-ruiz24-imperial-college-london/geom_phase_transition/<sweep_id>' `, 
    where `<sweep_id>` is will be found in the output message of the first step
- Check in wandb how the sweep is going! Also, you can kill the sweep (and hence the agents) from there
source venv/bin/activate; CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 wandb agent 'geom-phase-transitions/geom_phase_transitions/boq6300f'
>>>>>>> sergio
