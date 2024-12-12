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

match = re.search(r"Training completed. Results continuously saved to (?P<results_path>.+)", result.stdout)
results_path = match.group("results_path")  # results path is a json file

# ------------------ Test cases ------------------
# Load results
with open(results_path, "r") as f:
    result_json = json.load(f)
    config = result_json["298"][0]["config"]
    losses = result_json["298"][0]["losses"]["11"]

# Test config
assert config["embedding_dim"] == 4
assert config["depth"] == 2
assert config["num_params"] == 614

# Test losses
np.testing.assert_allclose(losses["train_loss"], 327.0042724609375, rtol=0.1)
np.testing.assert_allclose(losses["val_loss"], 355.8498229980469, rtol=0.1)
