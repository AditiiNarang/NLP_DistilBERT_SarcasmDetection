
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Training loss values shared in project summary
steps = [10, 40, 80, 100]
losses = [0.7389, 0.2686, 0.2509, 0.4021]

# ---------------------------------
# GRAPH 1: Training Loss Curve
# ---------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    steps, losses,
    marker="o",
    linewidth=2,
    markersize=7,
    label="Training Loss"
)

for x, y in zip(steps, losses):
    plt.annotate(
        f"{y:.4f}",
        (x, y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center"
    )

plt.title("DistilBERT Sarcasm Detection - Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.xticks(steps)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS / "training_loss_curve.png", dpi=300)
plt.show()

# ---------------------------------
# GRAPH 2: Loss Comparison
# ---------------------------------
plt.figure(figsize=(9, 5))

bars = plt.bar(
    [str(s) for s in steps],
    losses
)

for bar, loss in zip(bars, losses):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{loss:.4f}",
        ha="center",
        va="bottom"
    )

plt.title("Training Loss Comparison")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(RESULTS / "training_loss_comparison.png", dpi=300)
plt.show()

print("Graphs saved successfully in results/")
