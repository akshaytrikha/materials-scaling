import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def load_most_recent_results():
    """Load the most recent results file from the results directory."""
    results_dir = Path("results")
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        raise FileNotFoundError("No results files found in results directory")

    most_recent = max(result_files, key=lambda x: x.stat().st_mtime)
    with open(most_recent, "r") as f:
        return json.load(f)


def create_loss_plots(results):
    """Create vertically stacked train and validation loss plots with internal legends."""
    plt.style.use("seaborn-v0_8-paper")

    # Create figure with stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    # fig.suptitle('Training Progress Across Model Architectures', y=0.95)

    # Get param counts and sort models
    model_params = [(k, v["config"]["num_params"]) for k, v in results.items()]
    model_params.sort(key=lambda x: x[1])

    # Create color maps
    num_models = len(model_params)
    blues = LinearSegmentedColormap.from_list("", ["lightsteelblue", "navy"])(
        np.linspace(0, 1, num_models)
    )
    reds = LinearSegmentedColormap.from_list("", ["mistyrose", "darkred"])(
        np.linspace(0, 1, num_models)
    )

    # Plot for each model
    for idx, (model_key, num_params) in enumerate(model_params):
        model_data = results[model_key]

        # Extract losses and epochs
        epochs = list(map(int, model_data["losses"].keys()))
        train_losses = [d["train_loss"] for d in model_data["losses"].values()]
        val_losses = [d["val_loss"] for d in model_data["losses"].values()]

        # Create label with model parameters
        config = model_data["config"]
        breakpoint()
        if config["architecture"] == "FCN":
            label = f"{config['embedding_dim']}_h{config['hidden_dim']}_d{config['depth']} ({num_params:,} params)"
        elif config["architecture"] == "Transformer":
            label = (
                f"e{config['embedding_dim']}_d{config['depth']}_p{config['num_params']}"
            )

        # Plot with color gradients
        ax1.plot(
            epochs,
            train_losses,
            marker="o",
            markersize=4,
            label=label,
            color=blues[idx],
            linewidth=2,
        )
        ax2.plot(
            epochs,
            val_losses,
            marker="o",
            markersize=4,
            label=label,
            color=reds[idx],
            linewidth=2,
        )

    # Configure training loss subplot
    ax1.set_title("Training Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])  # Remove x-axis labels for top plot
    ax1.legend(
        loc="upper right",
        framealpha=0.9,  # Slightly transparent background
        edgecolor="black",  # Black edge for better visibility
        facecolor="white",
    )  # White background

    # Configure validation loss subplot
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor="black", facecolor="white")

    # Adjust layout and save
    plt.tight_layout()
    save_path = Path("results") / f"training_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    results = load_most_recent_results()
    create_loss_plots(results)
