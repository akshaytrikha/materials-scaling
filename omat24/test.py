import subprocess
import ast
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
        "--batch_size",
        "32",
    ],
    capture_output=True,
    text=True,
)

# ------------------ Extract stdout ------------------
stdout = result.stdout

# regex to match dictionaries in stdout
dict_pattern = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}")
dict_strings = dict_pattern.findall(stdout)

dicts = []
for ds in dict_strings:
    try:
        parsed_dict = ast.literal_eval(ds)
        dicts += [parsed_dict]
    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse dictionary: {ds}\nError: {e}")

# ------------------ Test cases ------------------
setup = dicts[0]
losses = dicts[1][0]

assert setup["architecture"] == "FCN"
assert setup["epochs"] == 1
assert setup["data_fraction"] == 0.01
assert setup["batch_size"] == 32

np.testing.assert_allclose(losses["train_loss"], 1713, rtol=0.1)
np.testing.assert_allclose(losses["val_loss"], 967, rtol=0.1)
