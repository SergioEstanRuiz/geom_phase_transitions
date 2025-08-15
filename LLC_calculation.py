import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
import pandas as pd
import os
from tqdm import tqdm
import wandb
from typing import List, Dict

# Import your configurations and models
import sys
sys.path.append('..')
from arithmetic_models import MLP, Transformer
from utils.produce_datasets import make_dataset, train_test_split
from configs import ExperimentParams, LLCParams

class LLCCalculator:
    def __init__(self):
        self.results_dir = "./results"
        self.llm_dir = "./results/EleutherAI"
        
    def find_arithmetic_models(self) -> Dict[str, str]:
        """Find trained arithmetic models"""
        models = {}
        if os.path.exists(self.results_dir):
            for exp_dir in os.listdir(self.results_dir):
                exp_path = os.path.join(self.results_dir, exp_dir)
                checkpoint_dir = os.path.join(exp_path, "checkpoints")
                params_file = os.path.join(exp_path, "params.csv")
                
                if os.path.isdir(exp_path) and os.path.exists(checkpoint_dir) and os.path.exists(params_file):
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                    if checkpoints:
                        models[exp_dir] = exp_path
        return models
    
    def find_llm_models(self) -> Dict[str, str]:
        """Find stored LLM models"""
        models = {}
        if os.path.exists(self.llm_dir):
            for model_dir in os.listdir(self.llm_dir):
                model_path = os.path.join(self.llm_dir, model_dir)
                if os.path.isdir(model_path):
                    models[f"EleutherAI/{model_dir}"] = model_path
        return models
    
    def display_models(self, arithmetic_models: Dict, llm_models: Dict):
        """Display available models"""
        print("\nAvailable Models:")
        print("-" * 50)
        
        all_models = {}
        idx = 1
        
        if arithmetic_models:
            print("Arithmetic Models:")
            for name, path in arithmetic_models.items():
                print(f"  {idx}. {name}")
                all_models[idx] = ('arithmetic', name, path)
                idx += 1
            print()
        
        if llm_models:
            print("Language Models:")
            for name, path in llm_models.items():
                print(f"  {idx}. {name}")
                all_models[idx] = ('llm', name, path)
                idx += 1
        
        print("-" * 50)
        return all_models
    
    def get_user_choice(self, all_models: Dict):
        """Get user's model selection"""
        while True:
            try:
                choice = int(input(f"\nSelect model (1-{len(all_models)}): "))
                if choice in all_models:
                    return all_models[choice]
                else:
                    print(f"Please enter a number between 1 and {len(all_models)}")
            except ValueError:
                print("Please enter a valid number")
    
    def get_llc_params(self) -> LLCParams:
        """Get LLC parameters"""
        default_params = LLCParams()
        
        print(f"\nDefault LLC Parameters:")
        print(f"  Learning Rate: {default_params.llc_lr}")
        print(f"  Chains: {default_params.num_chains}")
        print(f"  Draws: {default_params.num_draws}")
        
        use_default = input("\nUse default parameters? (y/n): ").lower() == 'y'
        
        if use_default:
            return default_params
        else:
            try:
                lr = input(f"Learning Rate [{default_params.llc_lr}]: ").strip()
                chains = input(f"Chains [{default_params.num_chains}]: ").strip()
                draws = input(f"Draws [{default_params.num_draws}]: ").strip()
                
                return LLCParams(
                    llc_lr=float(lr) if lr else default_params.llc_lr,
                    num_chains=int(chains) if chains else default_params.num_chains,
                    num_draws=int(draws) if draws else default_params.num_draws
                )
            except ValueError:
                print("Invalid input, using defaults")
                return default_params
    
    def setup_arithmetic_data(self, exp_path: str, batch_size: int):
        """Setup data for arithmetic models"""
        # Load experiment parameters
        params_df = pd.read_csv(os.path.join(exp_path, "params.csv"))
        params_dict = params_df.iloc[0].to_dict()
        
        # Create dataset
        dataset = make_dataset(int(params_dict['p']))
        train_data, _ = train_test_split(
            dataset, 
            params_dict['train_frac'], 
            int(params_dict['random_seed'])
        )
        
        return DataLoader(train_data, batch_size=batch_size, shuffle=True), params_dict
    
    def setup_llm_data(self, model_name: str, batch_size: int):
        """Setup data for LLM"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20%]")
        
        def tokenize_function(examples):
            encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
            encoding["labels"] = encoding["input_ids"].copy()
            return encoding
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    def evaluate_arithmetic(self, model, batch, device):
        """Evaluate arithmetic model"""
        model.eval()
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"logits": logits}
    
    def evaluate_llm(self, model, batch):
        """Evaluate LLM"""
        model.eval()
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device)
        }
        targets = batch["labels"].to(model.device)
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        return loss, {"logits": logits}
    
    def calculate_arithmetic_llc(self, exp_name: str, exp_path: str, llc_params: LLCParams):
        """Calculate LLC for arithmetic model"""
        print(f"\nCalculating LLC for arithmetic model: {exp_name}")
        
        # Setup
        data_loader, params_dict = self.setup_arithmetic_data(exp_path, llc_params.batch_size)
        
        # Load experiment parameters
        exp_params = ExperimentParams()
        for key, value in params_dict.items():
            if hasattr(exp_params, key):
                setattr(exp_params, key, value)
        
        # Get checkpoints FIRST
        checkpoint_dir = os.path.join(exp_path, "checkpoints")
        checkpoints = sorted([
            os.path.join(checkpoint_dir, f) 
            for f in os.listdir(checkpoint_dir) 
            if f.endswith('.pt')
        ], key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        # Get the epochs where we have checkpoints
        checkpoint_epochs = [int(cp.split('_')[-1].replace('.pt', '')) for cp in checkpoints]
        print(f"Checkpoint epochs: {checkpoint_epochs}")
        
        # Calculate LLC for ALL checkpoints first
        results = []
        llc_values = []
        
        for checkpoint_path in tqdm(checkpoints, desc="Processing checkpoints"):
            try:
                epoch = int(checkpoint_path.split('_')[-1].replace('.pt', ''))
                
                # Load model
                if exp_params.use_transformer:
                    model = Transformer(exp_params)
                else:
                    model = MLP(exp_params)
                
                model.load_state_dict(torch.load(checkpoint_path, map_location=llc_params.device))
                model = model.to(llc_params.device)
                
                # Calculate LLC
                llc_result = estimate_learning_coeff_with_summary(
                    model=model,
                    loader=data_loader,
                    evaluate=lambda m, b: self.evaluate_arithmetic(m, b, llc_params.device),
                    sampling_method=SGLD,
                    optimizer_kwargs=dict(
                        lr=llc_params.llc_lr,
                        nbeta=llc_params.llc_nbeta,
                        localization=llc_params.llc_localization
                    ),
                    num_chains=llc_params.num_chains,
                    num_draws=llc_params.num_draws,
                    device=llc_params.device,
                    online=False,
                )
                
                llc_value = llc_result['llc/mean']
                results.append((epoch, llc_value))
                llc_values.append(llc_value)
                
                tqdm.write(f"Epoch {epoch}: LLC = {llc_value:.4f}")
        
            except Exception as e:
                print(f"Error processing epoch {epoch}: {e}")
        
        # NOW log to wandb with aligned data
        if llc_params.log:
            config = {
                **params_dict,
                **llc_params.__dict__,
                "model_type": "Transformer" if exp_params.use_transformer else "MLP",
                "n_checkpoints": len(results)
            }
            
            run = wandb.init(
                project=exp_params.wandb_project,
                entity=exp_params.wandb_entity,
                name=f"llc_{exp_name}",
                config=config,
                tags=["arithmetic", "llc", exp_params.activation if hasattr(exp_params, 'activation') else "unknown"],
                settings=wandb.Settings(
                    _disable_stats=True,
                    _disable_meta=True,
                    console="off"
                )
            )
            
            # Load training data and filter to checkpoint epochs
            loss_file = os.path.join(exp_path, "loss_data.csv")
            if os.path.exists(loss_file):
                print("Loading and aligning training data...")
                training_data = pd.read_csv(loss_file)
                
                # Filter training data to only checkpoint epochs
                epoch_col = "epoch" if "epoch" in training_data.columns else "batch"
                training_data_filtered = training_data[training_data[epoch_col].isin(checkpoint_epochs)]
                
                print(f"Filtered training data from {len(training_data)} to {len(training_data_filtered)} points")
                
                # Create lists for plotting
                train_accuracies = training_data_filtered["train_acc"].tolist()
                train_losses = training_data_filtered["train_loss"].tolist()
                val_accuracies = training_data_filtered["val_acc"].tolist()
                val_losses = training_data_filtered["val_loss"].tolist()
                
                # Ensure same length as LLC data
                min_length = min(len(llc_values), len(train_accuracies))
                llc_values = llc_values[:min_length]
                train_accuracies = train_accuracies[:min_length]
                train_losses = train_losses[:min_length]
                val_accuracies = val_accuracies[:min_length]
                val_losses = val_losses[:min_length]
                checkpoint_epochs = checkpoint_epochs[:min_length]
                
                # Create plots like the working example
                wandb.log({
                    "accuracy_vs_llc": wandb.plot.line_series(
                        xs=[[i for i in range(min_length)] for _ in range(3)],
                        ys=[train_accuracies, val_accuracies, llc_values],
                        keys=["train_accuracy", "val_accuracy", "llc_mean"],
                        title=f"LLC vs Accuracy for {exp_name}",
                        xname="Checkpoint Index"
                    )
                })
                
                wandb.log({
                    "loss_vs_llc": wandb.plot.line_series(
                        xs=[[i for i in range(min_length)] for _ in range(3)],
                        ys=[train_losses, val_losses, llc_values],
                        keys=["train_loss", "val_loss", "llc_mean"],
                        title=f"LLC vs Loss for {exp_name}",
                        xname="Checkpoint Index"
                    )
                })
                
                # Log individual metrics (aligned)
                for i, (epoch, llc_val, train_acc, val_acc, train_loss, val_loss) in enumerate(
                    zip(checkpoint_epochs, llc_values, train_accuracies, val_accuracies, train_losses, val_losses)
                ):
                    wandb.log({
                        "checkpoint": i,
                        "epoch": epoch,
                        "llc_mean": llc_val,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })
            
            else:
                # If no training data, just log LLC
                for i, (epoch, llc_val) in enumerate(zip(checkpoint_epochs, llc_values)):
                    wandb.log({
                        "checkpoint": i,
                        "epoch": epoch,
                        "llc_mean": llc_val
                    })
            
            print(f"✅ Logged {len(llc_values)} aligned data points to wandb")
            wandb.finish()
        
        # Save results locally
        if results:
            df = pd.DataFrame(results, columns=['epoch', 'llc'])
            results_dir = f"./results/LLC/{exp_name}"
            os.makedirs(results_dir, exist_ok=True)
            df.to_csv(f"{results_dir}/llc_results.csv", index=False)
            print(f"Results saved to {results_dir}/llc_results.csv")
        
        return results
    
    def calculate_llm_llc(self, model_name: str, model_path: str, llc_params: LLCParams):
        """Calculate LLC for LLM"""
        print(f"\nCalculating LLC for LLM: {model_name}")
        
        # Setup data
        data_loader = self.setup_llm_data(model_name, llc_params.batch_size)
        
        # Initialize wandb if enabled
        if llc_params.log:
            # Use default ExperimentParams for wandb settings
            default_params = ExperimentParams()
            
            config = {
                **llc_params.__dict__,
                "model_name": model_name,
                "model_type": "LLM"
            }
            
            run = wandb.init(
                project=default_params.wandb_project,  # Use default settings
                entity=default_params.wandb_entity,    # Use default settings
                name=f"llc_{model_name.replace('/', '_')}",
                config=config,
                tags=["llm", "llc", model_name.split('/')[-1]]
            )
        
        # Define checkpoints to analyze
        checkpoints = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000]
        
        print(f"Will analyze {len(checkpoints)} checkpoints")
        
        # Calculate LLC for each checkpoint
        results = []
        for step in tqdm(checkpoints, desc="Processing checkpoints"):
            try:
                checkpoint_name = f"step{step}"
                
                # Load model at specific checkpoint
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    revision=checkpoint_name,
                    cache_dir=model_path
                ).to(llc_params.device)
                
                # Calculate LLC
                llc_result = estimate_learning_coeff_with_summary(
                    model=model,
                    loader=data_loader,
                    evaluate=self.evaluate_llm,
                    sampling_method=SGLD,
                    optimizer_kwargs=dict(
                        lr=llc_params.llc_lr,
                        nbeta=llc_params.llc_nbeta,
                        localization=llc_params.llc_localization
                    ),
                    num_chains=llc_params.num_chains,
                    num_draws=llc_params.num_draws,
                    device=llc_params.device,
                    online=False,
                )
                
                llc_value = llc_result['llc/mean']
                results.append((step, llc_value))
                
                # Log to wandb
                if llc_params.log:
                    wandb.log({"llc": llc_value, "training_step": step}, step=step)
                    tqdm.write(f"Step {step}: LLC = {llc_value:.4f}")
            
            except Exception as e:
                print(f"Error processing step {step}: {e}")
        
        # Save results
        if results:
            df = pd.DataFrame(results, columns=['step', 'llc'])
            model_safe_name = model_name.replace('/', '_')
            results_dir = f"./results/LLC/{model_safe_name}"
            os.makedirs(results_dir, exist_ok=True)
            df.to_csv(f"{results_dir}/llc_results.csv", index=False)
            print(f"Results saved to {results_dir}/llc_results.csv")
        
        # Finish wandb run
        if llc_params.log:
            wandb.finish()
    
        return results

def main():
    calculator = LLCCalculator()
    
    print("Learning Coefficient Calculator")
    print("=" * 40)
    
    # Find available models
    arithmetic_models = calculator.find_arithmetic_models()
    llm_models = calculator.find_llm_models()
    
    if not arithmetic_models and not llm_models:
        print("No models found!")
        print("- Train arithmetic models using train.py")
        print("- Place LLM models in results/EleutherAI/")
        return
    
    # Display and get selection
    all_models = calculator.display_models(arithmetic_models, llm_models)
    model_type, model_name, model_path = calculator.get_user_choice(all_models)
    
    # Get LLC parameters
    llc_params = calculator.get_llc_params()
    
    # Calculate LLC based on model type
    if model_type == 'arithmetic':
        results = calculator.calculate_arithmetic_llc(model_name, model_path, llc_params)
    else:
        results = calculator.calculate_llm_llc(model_name, model_path, llc_params)
    
    print(f"\nCompleted! Processed {len(results)} checkpoints.")

if __name__ == "__main__":
    main()