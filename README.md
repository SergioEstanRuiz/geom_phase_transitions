### Quick intro 
The aim of this repo is to study geometric phase transitions in models during training, through the lens of singular learning theory. To this end, we are looking at a variety of models and phenomena, including:
- Grokking in modular arithmetic models
- Stagewise Development in LLMs

### How this repo is structured: 
Honestly, it's still a bit of a mess. We'll get it sorted it! Some information:
- The idea is for `main.py` to control everything. Right now it only trains models. We could have it so that it can train models and then calculates LLC curves. This file calls auxiliary scripts like `training.py`, which controls how the training of the model is done, or `arithmetic_models.py`, where the different models used in the arithmetic task are placed, or `produce_datasets.py` which creates the modular arithmetic datasets. 
- On the other hand, we have the `visualising_results.ipynb` which right now does the LLC calculations and then plots curves. 
