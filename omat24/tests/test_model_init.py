import warnings
import json
from collections import defaultdict

warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")
# External
import torch
from pathlib import Path

# Internal
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS
from models.fcn import FCNModel
from models.transformer_models import XTransformerModel
from models.schnet import SchNet
from models.equiformer_v2 import EquiformerS2EFS
from train_utils import run_validation

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

# Configuration dictionary for different model sizes
MODEL_CONFIGS = {
    "FCN": {
        "1k": {
            "vocab_size": 119,
            "embedding_dim": 12,
            "hidden_dim": 12,
            "depth": 2,
            "use_factorized": False,
        },
        "50k": {
            "vocab_size": 119,
            "embedding_dim": 64,
            "hidden_dim": 64,
            "depth": 8,
            "use_factorized": False,
        },
    },
    "Transformer": {
        "1k": {
            "num_tokens": 119,
            "d_model": 1,
            "depth": 1,
            "n_heads": 1,
            "d_ff_mult": 1,
            "use_factorized": False,
        },
        "50k": {
            "num_tokens": 119,
            "d_model": 32,
            "depth": 2,
            "n_heads": 1,
            "d_ff_mult": 4,
            "use_factorized": False,
        },
    },
    "SchNet": {
        "1k": {
            "hidden_channels": 6,
            "num_filters": 6,
            "num_interactions": 2,
            "num_gaussians": 6,
        },
        "50k": {
            "hidden_channels": 48,
            "num_filters": 32,
            "num_interactions": 6,
            "num_gaussians": 32,
        },
    },
}


def run_test():
    # args = get_args()
    datasets = VALID_DATASETS
    split_name = "val"
    val_data_fraction = 0.01
    datasets_base_path = "./datasets"
    n_elements = 119

    # Download datasets if not present
    dataset_paths = []
    for dataset_name in datasets:
        dataset_path = Path(f"{datasets_base_path}/{split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, split_name, datasets_base_path)
        dataset_paths.append(dataset_path)

    # User Hyperparam Feedback
    batch_size = 64
    use_factorize = False

    # Results dictionary to store validation losses
    results = defaultdict(dict)

    # Iterate over architectures and parameter scales
    for architecture in ["FCN", "SchNet", "Transformer"]:
        graph = architecture in ["SchNet", "EquiformerV2"]

        # Determine max_n_atoms for Transformer if needed
        if architecture == "Transformer":
            if split_name == "train":
                max_n_atoms = 236
            elif split_name == "val":
                max_n_atoms = 168

        for param_scale in ["1k", "50k"]:
            config = MODEL_CONFIGS[architecture][param_scale]

            # Initialize model with appropriate config
            if architecture == "FCN":
                model = FCNModel(**config)
            elif architecture == "Transformer":
                model = XTransformerModel(**config)
            elif architecture == "SchNet":
                model = SchNet(**config)

            print(
                f"\n{architecture} {param_scale} Model is on device {DEVICE} and has {model.num_params} parameters"
            )

            # Get data loaders
            _, val_loader = get_dataloaders(
                dataset_paths,
                train_data_fraction=0,
                batch_size=batch_size,
                seed=SEED,
                architecture=architecture,
                batch_padded=False,
                val_data_fraction=val_data_fraction,
                train_workers=0,
                val_workers=0,
                graph=graph,
                factorize=use_factorize,
            )

            dataset_size = len(val_loader.dataset)
            print(f"graph: {graph}")
            print(
                f"\nValidating on dataset fraction {val_data_fraction} with {dataset_size} samples"
            )

            # Run validation
            (
                val_loss,
                val_energy_loss,
                val_force_loss,
                val_stress_iso_loss,
                val_stress_aniso_loss,
            ) = run_validation(
                model=model,
                val_loader=val_loader,
                graph=graph,
                device=DEVICE,
                factorize=False,
                use_mixed_precision=False,
            )

            # Store results
            results[architecture][param_scale] = {
                "params": model.num_params,
                "val_loss": val_loss,
                "val_energy_loss": val_energy_loss,
                "val_force_loss": val_force_loss,
                "val_stress_iso_loss": val_stress_iso_loss,
                "val_stress_aniso_loss": val_stress_aniso_loss,
            }

            print(
                f"\n{architecture} {param_scale} Model had initial val_loss of {val_loss}"
            )

    # Print summary table of results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Architecture':<15} {'Size':<10} {'Parameters':<15} {'Val Loss':<15}")
    print("-" * 80)

    for arch in results:
        for size in results[arch]:
            print(
                f"{arch:<15} {size:<10} {results[arch][size]['params']:<15} {results[arch][size]['val_loss']:<15.6f}"
            )

    # # Save results to file
    # with open("model_comparison_results.json", "w") as f:
    #     json.dump(results, f, indent=2)

    # print("\nResults saved to model_comparison_results.json")


if __name__ == "__main__":
    run_test()
