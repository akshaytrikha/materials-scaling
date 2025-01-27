# External
import subprocess
import json
import re
import numpy as np
import os


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
        "0.001",
    ],
    capture_output=True,
    text=True,
)

match = re.search(r"Results continuously saved to (?P<results_path>.+)", result.stdout)
results_path = match.group("results_path")  # results path is a json file

# ------------------ Test cases ------------------
# Load results
with open(results_path, "r") as f:
    result_json = json.load(f)
    config = result_json["373"][0]["config"]
    losses = result_json["373"][0]["losses"]["6"]

# Test config
assert config["embedding_dim"] == 4
assert config["depth"] == 2
assert config["num_params"] == 614

# Test losses
try:
    np.testing.assert_allclose(losses["train_loss"], 163.54835764567056, rtol=0.1)
    np.testing.assert_allclose(losses["val_loss"], 157.8370106362889, rtol=0.1)
except AssertionError as e:
    print(e)

# Delete results file
os.remove(results_path)
