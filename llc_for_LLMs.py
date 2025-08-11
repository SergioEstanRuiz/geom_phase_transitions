from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from devinterp.optim.sgld import SGLD
import torch.nn.functional as F
from dataclasses import dataclass
import os
import pandas as pd
import wandb

# PARAMETERS - to be cleaned up
@dataclass
class Params:
    model_name: str = "EleutherAI/pythia-70m"  # replace size as needed
    batch_size: int = 8
    llc_lr : float = 0.003
    llc_nbeta: float = 2.0
    llc_localization: float = 5.0
    num_chains: int = 2
    num_draws: int = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log: bool = True

params = Params()

# Upload pythia model and tokenizer
checkpoint_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
checkpoint_list += [i * 1000 for i in range(2, 101)]

tokenizer = AutoTokenizer.from_pretrained(params.model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to be able to tokenize with padding

# PREPARE DATALOADER FOR LLC CALCULATION
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20%]") # Load the wikitext-2 dataset

# tokenize the dataset
def tokenize_function(examples):
    encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# create a DataLoader
data_loader = DataLoader(tokenized_dataset, batch_size=params.batch_size, shuffle=True)

# EVALUATE MODEL for LLC
def evaluate(model, batch):
    model.eval()

    inputs = {
        "input_ids": batch["input_ids"].to(model.device),
        "attention_mask": batch["attention_mask"].to(model.device)
    } # Exclude labels from inputs
    targets = batch["labels"].to(model.device) # Get labels (ie. targets) and move to device

    outputs = model(**inputs) # Forward pass through the model, ** unpacks the dictionary inputs {text: ..., attention_mask: ...}
    logits = outputs.logits # Gives the raw logits from the model, ie. [batch, seq, vocab]
                            # For some batch b, and point of our setence t, logits[b, t, :] gives the logits for each token in the vocabulary
                            # ie. the bigger logit[b, t, i] is, the more likely the model thinks token i is the next token in the sequence

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # [batch*seq, vocab]
        targets.view(-1)                   # [batch*seq], ie. flatten the targets
    )

    return loss, {
        "logits": logits,
    }

if params.log:
    run = wandb.init(project="llc_for_llms",config=params)

llcs = []
os.makedirs(f"./results/{params.model_name}", exist_ok=True)
for checkpoint in tqdm(checkpoint_list):
    checkpoint = f"step{checkpoint}"
    model = AutoModelForCausalLM.from_pretrained(params.model_name, revision=checkpoint).to(params.device)
    llc = estimate_learning_coeff_with_summary(
    model=model,
    loader=data_loader,
    evaluate=evaluate,
    sampling_method=SGLD,
    optimizer_kwargs=dict(lr=params.llc_lr, nbeta=params.llc_nbeta, localization=params.llc_localization),
    num_chains=params.num_chains,
    num_draws=params.num_draws,
    device=params.device,
    online=False,
    )['llc/mean']
    if params.log:
        run.log({"checkpoint": checkpoint,
                  "llc": llc})
    llcs.append((checkpoint, llc))

# Save the results
llcs_df = pd.DataFrame(llcs, columns=["revision", "llc"])
llcs_df.to_csv(f"./results/{params.model_name}/llc_results.csv", index=False)