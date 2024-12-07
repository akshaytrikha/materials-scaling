import json
import numpy as np

np.random.seed(42)

dataset_sizes = [1e3, 1e4, 1e5]
param_options = [1e4, 5e4, 1e5, 5e5, 1e6]

# Let's define a function that simulates saturation:
# loss ~ (C * ds_size^(-alpha)) + model_contribution
# where model_contribution might decrease initially with model size,
# but approaches a floor value.

C = 200.0
alpha = 0.15


def simulate_loss(ds_size, params):
    # Base loss decreases with ds_size
    base_loss = C * (ds_size ** (-alpha))
    # Model effect: bigger models do better up to a point:
    # Let's say loss improvement ~ (params^-beta) but saturates after some threshold
    beta = 0.2
    # Saturation point: after large params, improvement is minimal
    # We'll implement a soft saturation using a logistic function
    max_improvement = 0.5 * base_loss  # max improvement is half of base_loss
    scaled_params = params / 1e5
    improvement = max_improvement * (scaled_params**beta) / (1 + scaled_params**beta)
    # But improvement reduces the loss, so final:
    # Add a small noise component
    final_loss = base_loss - improvement + np.random.normal(scale=0.01 * base_loss)
    return final_loss


all_results = {}
for ds_size in dataset_sizes:
    runs = []
    for p in param_options:
        # We'll record a few training steps, though we focus on the best val_loss
        # Steps from a higher initial loss down to final_loss:
        final_loss = simulate_loss(ds_size, p)
        current_loss = final_loss * 1.5
        steps = {}
        for step in range(5):
            # Simulate progressive improvement
            current_loss = current_loss - (current_loss - final_loss) / 2.0
            steps[str(step)] = {
                "train_loss": current_loss
                + np.random.normal(scale=0.02 * current_loss),
                "val_loss": current_loss + np.random.normal(scale=0.02 * current_loss),
            }
        run_dict = {
            "model_name": f"model_ds{int(ds_size)}_p{int(p)}",
            "config": {
                "architecture": "Transformer",
                "embedding_dim": 8,
                "depth": 2,
                "num_params": p,
                "dataset_size": ds_size,
            },
            "losses": steps,
            "checkpoint_path": f"checkpoints/Transformer_ds{int(ds_size)}_p{int(p)}.pth",
        }
        runs.append(run_dict)
    all_results[int(ds_size)] = runs

with open("dummy_experiments_saturation.json", "w") as f:
    json.dump(all_results, f, indent=4)
