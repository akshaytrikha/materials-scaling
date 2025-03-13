import os

filenames_to_include = [
    "data.py",
    "data_utils.py",
    "loss.py",
    "models/equiformer_v2.py",
    "models/fcn.py",
    "models/transformer_models.py",
    "models/schnet.py",
    "models/model_utils.py",
    "train.py",
    "train_utils.py",
    "log_utils.py",
    "arg_parser.py",
    "tests/test_dataloader.py",
    "tests/test_fcn.py",
    "tests/test_transformer.py",
    "tests/test_schnet.py",
    "tests/test_equiformer_v2.py",
    "tests/test_model_init.py",
    "tests/test_model_prediction_evolution.py",
]
directory_path = "."
output_file = "context.txt"


def include_file_contents(filenames, directory=".", output_file="context.txt"):
    """Reads the content of the specified files in a directory and writes them into an output file."""
    with open(output_file, "w") as output:
        for filename in filenames:
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r") as file:
                    content = file.read()
                output.write(f"# {filename}\n{content}\n")
            else:
                print(f"File not found: {filepath}")


include_file_contents(filenames_to_include, directory_path, output_file)
