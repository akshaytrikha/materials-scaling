import subprocess
import json
import re
import numpy as np


# ------------------ Run small training job ------------------
result = subprocess.run(
    [
        "python3",
        "train.py",
        "--architecture",
        "FCN",
        "--epochs",
        "1",
        "--data_fraction",
        "0.01",
        "--batch_sizes",
        "32",
        "--lrs",
        "0.001"
    ],
    capture_output=True,
    text=True
)
print(result.stderr)

match = re.search(r"Results saved to (?P<results_path>.+)", result.stdout)
results_path = match.group("results_path")  # results path is a json file

# ------------------ Test cases ------------------
# Load results
with open(results_path, "r") as f:
    result_json = json.load(f)
    config = result_json["model_0_batch_size_32_lr_0.001"]["config"]
    losses = result_json["model_0_batch_size_32_lr_0.001"]["losses"]["0"]

# Test config
assert config["embedding_dim"] == 32
assert config["depth"] == 2
assert config["num_params"] == 15338

# Test losses
np.testing.assert_allclose(losses["train_loss"], 386.99095837, rtol=0.1)
np.testing.assert_allclose(losses["val_loss"], 282.811146, rtol=0.1)
