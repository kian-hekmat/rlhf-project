import subprocess
import sys

try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb", "-q"])
    import wandb

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt

import pandas as pd

api = wandb.Api()

run = api.run("michael_li-university-of-california-berkeley/llm-rl-final-project/bvs10h92")
history = run.history(pandas=True)

print("Available columns:", [c for c in history.columns if "accuracy" in c.lower() or "loss" in c.lower()])
print(history.head(5))

history.to_csv("reward_model_history.csv", index=False)
print("Saved reward_model_history.csv")

train_acc_col = next((c for c in history.columns if "pair_accuracy" in c and "eval" not in c), None)
eval_acc_col = next((c for c in history.columns if "pair_accuracy" in c and "eval" in c), None)
loss_col = next((c for c in history.columns if "loss" in c.lower() and "eval" not in c), None)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if train_acc_col:
    data = history[["_step", train_acc_col]].dropna()
    axes[0].plot(data["_step"], data[train_acc_col], label="Train", color="steelblue", linewidth=2)
if eval_acc_col:
    data = history[["_step", eval_acc_col]].dropna()
    axes[0].plot(data["_step"], data[eval_acc_col], label="Eval", color="darkorange", linewidth=2, linestyle="--")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Pair Accuracy")
axes[0].set_title("Reward Model Pair Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if loss_col:
    data = history[["_step", loss_col]].dropna()
    axes[1].plot(data["_step"], data[loss_col], color="crimson", linewidth=2)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Reward Model Training Loss")
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_model_training_curve.png", dpi=150, bbox_inches="tight")
print("Saved reward_model_training_curve.png")
plt.show()
