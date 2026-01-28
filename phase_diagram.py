import math
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

ENTITY = "riya-danait01-university-of-oxford"
PROJECT = "geom_phase_transitions"
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.exp_name": "sweep_train-frac"})
records = []

for run in runs:
    cfg = dict(run.config)
    summary = dict(run.summary)

    M = cfg.get("derived/M", cfg.get("p"))
    N = cfg.get("derived/N")
    train_frac = cfg.get("train_frac")
    N_over_M2 = cfg.get("derived/N_over_M2")
    N_over_MlogM = cfg.get("derived/N_over_MlogM")

    rec = {
        "run_id": run.id, 
        "name": run.name, 
        "M": M, 
        "N": N, 
        "N_over_M2": N_over_M2, 
        "N_over_MlogM": N_over_MlogM, 
        "regime": summary.get("dyn/regime"), 
        "train_frac": train_frac,
        "final_val_acc": summary.get("dyn/final_val_acc", summary.get("val/acc", None)),
        "final_train_acc": summary.get("dyn/final_train_acc"), 
        "grokking_test": summary.get("grokking_test")
    }
    records.append(rec)

df = pd.DataFrame(records)
print("Loaded runs:", len(df))
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

fig, ax = plt.subplots(figsize=(6, 4))
for regime, sub in df.groupby("regime"):
    ax.scatter(
        sub["N_over_MlogM"], 
        sub["final_val_acc"], 
        label=str(regime), 
        alpha=0.7, 
        s=30
    )
    for i, (_,row) in enumerate(df.iterrows()):
        x = row["N_over_MlogM"]
        y = row["final_val_acc"]
        tf = row["train_frac"]

        if i < 2:
            ax.text(
            x, y, f"  train_frac: {tf:.2f}",
            ha="left", va="bottom",
            rotation=35,
            rotation_mode="anchor",
            fontsize=8, alpha=0.9
            )
        elif (i >= 2 and i <= 3):
            ax.text(
            x, y, f"  train_frac: {tf:.2f}",
            ha="left", va="top",
            rotation=-35,
            rotation_mode="anchor",
            fontsize=8, alpha=0.9
            )
        else:
            ax.text(
            x, y, f"  train_frac: {tf:.2f}  ",
            ha="right", va="top",
            rotation=35,
            rotation_mode="anchor",
            fontsize=8, alpha=0.9
            )

ax.set_xlabel(r"$N\, / \,(M\,\log M)$")
ax.set_ylabel("Final Validation Accuracy")
ax.legend()
ax.grid(True, alpha=0.2)
plt.title(f"Phase Diagram in Scaled Data Size, $M$ = {M}")
plt.tight_layout()
plt.show()

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.exp_name": "sweep_p_train-frac"})
records = []

for run in runs:
    cfg = dict(run.config)
    summary = dict(run.summary)

    M = cfg.get("derived/M", cfg.get("p"))
    N = cfg.get("derived/N")
    train_frac = cfg.get("train_frac")
    N_over_M2 = cfg.get("derived/N_over_M2")
    N_over_MlogM = cfg.get("derived/N_over_MlogM")

    rec = {
        "run_id": run.id, 
        "name": run.name, 
        "M": M, 
        "N": N, 
        "N_over_M2": N_over_M2, 
        "N_over_MlogM": N_over_MlogM, 
        "regime": summary.get("dyn/regime"), 
        "train_frac": train_frac,
        "max_gap": summary.get("dyn/max_gap"),
        "final_val_acc": summary.get("dyn/final_val_acc", summary.get("val/acc", None)),
        "final_train_acc": summary.get("dyn/final_train_acc"), 
        "grokking_test": summary.get("grokking_test")
    }
    records.append(rec)

df = pd.DataFrame(records)
print("Loaded runs:", len(df))
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

grouped = (df.groupby(["M", "N_over_MlogM"]).agg(
    mean_val_acc=("final_val_acc", "mean"),
    num_runs=("final_val_acc", "size")
).reset_index())

#print(tabulate(grouped, headers="keys", tablefmt="psql", showindex=False))
line_handles = []
color_for_M = {}
fig, ax = plt.subplots(figsize=(7, 5))
for M, sub in grouped.groupby("M"):

    sub_sorted = sub.sort_values("N_over_MlogM")
    x = sub_sorted["N_over_MlogM"].values
    y = sub_sorted["mean_val_acc"].values
    
    (line,) = ax.plot(x, y, linewidth=1.5, alpha=0.8, label=f"M={int(M)}")
    line_handles.append(line)
    color_for_M[M] = line.get_color()

regime_markers = {
    "no_generalization": "x",
    "grokking": "o",
    "immediate_generalization": "s"
}
scatter_handles = {}
for (M, regime), sub in df.groupby(["M", "regime"]):
    marker = regime_markers.get(regime, "o")
    color = color_for_M.get(M, "gray")
    h = ax.scatter(
        sub["N_over_MlogM"], 
        sub["final_val_acc"], 
        label=str(regime), 
        alpha=0.9, 
        s=40,
        color=color, 
        marker=marker,
        edgecolor="black",
        linewidths=0.5
    )
    scatter_handles[regime] = h
    
ax.set_xlabel(r"$N\, / \,(M\,\log M)$")
ax.set_ylabel("Final Validation Accuracy")
ax.grid(True, alpha=0.2)

legend1 = ax.legend(
    handles = line_handles,
    title = r"Group size $M$",
    loc = "lower right",
    fontsize = 8
)
ax.add_artist(legend1)

regime_labels = {
    "no_generalization": "no_generalization",
    "grokking": "grokking",
    "immediate_generalization": "immediate_generalization"
}
legend2 = ax.legend(
    handles=[scatter_handles[r] for r in scatter_handles.keys()],
    labels=[regime_labels[r] for r in scatter_handles.keys()],
    title="Regime",
    loc="center right",
    fontsize=8
)
plt.title(r"Scaling Curves: Val Acc versus $N\, / \,(M\,\log M)$ for various $M$")
plt.tight_layout()
plt.show()