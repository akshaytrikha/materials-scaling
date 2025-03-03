# External
import torch
import argparse
from pathlib import Path
import json
import yaml
import numpy as np
from tqdm import tqdm
import requests
import warnings
from fairchem.core.common.registry import Registry
from fairchem.core.modules.evaluator import Evaluator

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch_geometric.deprecation"
)

# Internal
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS
from train_utils import forward_pass, collect_samples_helper
from models.fcn import FCNModel
from models.transformer_models import XTransformerModel
from models.schnet import SchNet
from models.equiformer_v2 import EquiformerS2EF

# Set seed & device
SEED = 1024
DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/EquiformerV2_ds1008140_p1032742_20250301_085158.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["rattled-300-subsampled"],
        help="Dataset(s) to use",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="Fraction of the dataset to use for validation",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--datasets_base_path",
        type=str,
        default="./datasets",
        help="Base path for dataset storage",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_samples", type=int, default=0, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--fairchem_model",
        type=str,
        default=None,
        choices=["eqV2_31M", "eqV2_86M", "eqV2_153M"],
        help="Name of the FAIRChem model config to use (if using FAIRChem models)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate used during training (for visualization metadata)",
    )
    return parser.parse_args()


def fetch_fairchem_eqV2_config(config_name):
    """Fetch FAIRChem's EquiformerV2 config from GitHub and save it locally.

    Args:
        config_name (str): Name of the config to fetch (eqV2_31M, eqV2_86M, or eqV2_153M)

    Returns:
        tuple: (config_dict, output_path) - The loaded config as a dictionary and the path where it was saved
    """
    # Create configs directory if it doesn't exist
    configs_dir = Path("fairchem_configs")
    configs_dir.mkdir(exist_ok=True)

    # Determine output path based on config name
    output_path = configs_dir / f"{config_name}_config.yml"

    # Check if file already exists
    if output_path.exists():
        print(f"Config file {output_path} already exists, loading from disk.")
        with open(output_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Config loaded from {output_path}")
        return config, output_path

    # Map config names to their URLs
    config_urls = {
        "eqV2_31M": "https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/configs/omat24/mptrj/eqV2_31M_mptrj.yml",
        "eqV2_86M": "https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/configs/omat24/all/eqV2_86M.yml",
        "eqV2_153M": "https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/configs/omat24/all/eqV2_153M.yml",
    }

    if config_name not in config_urls:
        raise ValueError(
            f"Unknown config name: {config_name}. Available options: {list(config_urls.keys())}"
        )

    # Get the URL for the specified config
    config_url = config_urls[config_name]

    try:
        # If it's a GitHub UI URL, convert to raw format
        if "github.com" in config_url and "/blob/" in config_url:
            config_url = config_url.replace(
                "github.com", "raw.githubusercontent.com"
            ).replace("/blob/", "/")

        # Fetch the config
        response = requests.get(config_url)
        response.raise_for_status()  # Ensure the request was successful

        # Load the YAML
        config = yaml.safe_load(response.text)

        # Save to disk
        with open(output_path, "w") as f:
            yaml.dump(config, f)

        print(f"{config['model']['name']} config loaded and saved to {output_path}")
        return config, output_path

    except (requests.RequestException, yaml.YAMLError) as e:
        print(f"Error fetching or parsing config: {e}")
        raise


def load_fairchem_eqV2(config_name, device):
    """Load a pretrained FAIRChem EquiformerV2 model.

    Args:
        config_name (str): Name of the config to use
        device (torch.device): Device to load the model on

    Returns:
        tuple: (model, architecture) where model is the loaded model and architecture is the model name
    """
    # Fetch the config
    config, config_path = fetch_fairchem_eqV2_config(config_name)

    # Get the model configuration
    model_config = config["model"]

    # Remove the stress head to avoid the error with rank2_symmetric_head if present
    if "heads" in model_config and "stress" in model_config["heads"]:
        print(
            "Warning: Removing stress head as 'rank2_symmetric_head' is not registered in the fairchem registry"
        )
        del model_config["heads"]["stress"]

    try:
        # Get the model class from FAIRChem's registry
        ModelClass = Registry.get_model_class(
            model_config["name"]
        )  # e.g., "hydra" -> HydraModel class

        # Instantiate the model with backbone and other parameters from the config
        model = ModelClass(
            backbone=model_config.get("backbone"),
            heads=model_config.get("heads"),
            otf_graph=model_config.get("otf_graph", True),
            pass_through_head_outputs=model_config.get(
                "pass_through_head_outputs", False
            ),
        )

        # Set model name and architecture
        model.name = f"FAIREquiformerV2"
        architecture = "EquiformerV2"

        # Load weights from local path
        local_weights_path = "eqV2_31M_omat.pt"
        print(f"Loading model weights from {local_weights_path}")

        # Load the state dictionary
        weights = torch.load(local_weights_path, map_location=device)

        # Check if the loaded file contains 'model_state_dict' key (checkpoint format)
        if isinstance(weights, dict) and "model_state_dict" in weights:
            weights = weights["model_state_dict"]

        # Load the weights into the model
        model.load_state_dict(weights, strict=True)
        print(f"Successfully loaded weights from {local_weights_path}")

        # Move model to device
        model.to(device)
        model.eval()

        print(f"Loaded FAIRChem {architecture} model")
        return model, architecture

    except Exception as e:
        print(f"Error loading FAIRChem model: {e}")
        raise


def load_checkpoint_model(checkpoint_path, device):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract architecture from checkpoint filename
    checkpoint_name = Path(checkpoint_path).stem
    parts = checkpoint_name.split("_")
    architecture = parts[0]

    # Find parameter count from filename (after 'p')
    param_count = None
    for part in parts:
        if part.startswith("p"):
            param_count = int(part[1:])
            break

    if param_count is None:
        raise ValueError(
            f"Could not determine parameter count from checkpoint name {checkpoint_name}"
        )

    print(f"Loading {architecture} model with {param_count} parameters")

    # Create the appropriate model
    if architecture == "FCN":
        model = FCNModel(
            vocab_size=119,
            embedding_dim=128,  # Default values, will be overridden by state dict
            hidden_dim=256,  # Default values, will be overridden by state dict
            depth=4,  # Default values, will be overridden by state dict
            use_factorized=False,
        )
    elif architecture == "Transformer":
        model = XTransformerModel(
            num_tokens=119,
            d_model=64,  # Default values, will be overridden by state dict
            depth=6,  # Default values, will be overridden by state dict
            n_heads=8,  # Default values, will be overridden by state dict
            d_ff_mult=16,  # Default values, will be overridden by state dict
            use_factorized=False,
        )
    elif architecture == "SchNet":
        model = SchNet(
            hidden_channels=128,  # Default values, will be overridden by state dict
            num_filters=128,  # Default values, will be overridden by state dict
            num_interactions=6,  # Default values, will be overridden by state dict
            num_gaussians=50,  # Default values, will be overridden by state dict
            device=device,
        )
    elif architecture == "EquiformerV2":
        # # 1,032,742 params
        config = {
            "regress_forces": True,
            "use_pbc": True,
            "use_pbc_single": True,
            "otf_graph": True,
            "enforce_max_neighbors_strictly": False,
            "max_neighbors": 20,
            "max_radius": 12.0,
            "max_num_elements": 119,
            "num_layers": 3,
            "sphere_channels": 36,
            "attn_hidden_channels": 36,
            "num_heads": 4,
            "attn_alpha_channels": 36,
            "attn_value_channels": 12,
            "ffn_hidden_channels": 72,
            "norm_type": "layer_norm_sh",
            "lmax_list": [2],
            "mmax_list": [1],
            "grid_resolution": 14,
            "num_sphere_samples": 64,
            "edge_channels": 72,
            "use_atom_edge_embedding": True,
            "share_atom_edge_embedding": False,
            "use_m_share_rad": False,
            "distance_function": "gaussian",
            "num_distance_basis": 144,
            "attn_activation": "silu",
            "use_s2_act_attn": False,
            "use_attn_renorm": True,
            "ffn_activation": "silu",
            "use_gate_act": False,
            "use_grid_mlp": True,
            "use_sep_s2_act": True,
            "alpha_drop": 0.1,
            "drop_path_rate": 0.1,
            "proj_drop": 0.0,
            "weight_init": "uniform",
        }
        model = EquiformerS2EF(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Load the state dict into the model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, architecture, param_count


def compute_metrics(loader, model, device, graph=False, factorize=False):
    """Compute evaluation metrics using fairchem's Evaluator.

    Args:
        loader: DataLoader for the evaluation data
        model: The model to evaluate
        device: The device to run evaluation on
        graph: Whether using graph-based models
        factorize: Whether using factorized distances

    Returns:
        dict: Metrics dictionary
    """
    model.eval()

    # Configure the evaluator with the desired metrics
    eval_config = {
        "energy": ["mae"],
        "forces": [
            "mae",
            "forcesx_mae",
            "forcesy_mae",
            "forcesz_mae",
            "cosine_similarity",
        ],
        "stress": ["mae"],
    }

    # Initialize the FAIRChem evaluator
    evaluator = Evaluator(task="s2ef", eval_metrics=eval_config)

    # Keep track of metrics across batches
    metrics = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing metrics"):
            # Forward pass
            (
                pred_forces,
                pred_energy,
                pred_stress,
                true_forces,
                true_energy,
                true_stress,
                mask,
                natoms,
            ) = forward_pass(
                model=model,
                batch=batch,
                graph=graph,
                training=False,
                device=device,
                factorize=factorize,
            )

            # For non-graph models with padding, we need to apply the mask
            if not graph and mask is not None:
                # Create a flat index of valid atoms
                batch_size = pred_forces.shape[0]
                valid_forces_pred = []
                valid_forces_target = []

                # Process each structure in the batch
                for i in range(batch_size):
                    # Select only valid atoms based on mask
                    valid_mask = mask[i].bool()
                    valid_forces_pred.append(pred_forces[i, valid_mask])
                    valid_forces_target.append(true_forces[i, valid_mask])

                # Concatenate all valid forces
                pred_forces = torch.cat(valid_forces_pred, dim=0)
                true_forces = torch.cat(valid_forces_target, dim=0)

            # Prepare the predictions and targets in the format expected by FAIRChem's evaluator
            predictions = {
                "energy": pred_energy,
                "forces": pred_forces,
                "stress": pred_stress,
                "natoms": natoms,
            }

            targets = {
                "energy": true_energy,
                "forces": true_forces,
                "stress": true_stress,
                "natoms": natoms,
            }

            # Update metrics for this batch
            metrics = evaluator.eval(predictions, targets, prev_metrics=metrics)

    # Extract the final metrics
    final_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict) and "metric" in value:
            final_metrics[key] = value["metric"]

    # Convert to meV if needed (multiplying by 1000)
    for key in final_metrics:
        if key.startswith(
            ("energy", "forces", "forcesx", "forcesy", "forcesz", "stress")
        ) and not key.endswith(("cosine_similarity")):
            final_metrics[key] *= 1000  # Convert from eV to meV

    return final_metrics


def main():
    # Parse arguments
    args = parse_args()
    output_dir = Path("inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model - either from checkpoint or FAIRChem
    if args.fairchem_model:
        # Load FAIRChem model
        model, architecture = load_fairchem_eqV2(args.fairchem_model, DEVICE)
        # Extract parameter count from model
        param_count = sum(p.numel() for p in model.parameters())
    else:
        # Load model from checkpoint
        model, architecture, param_count = load_checkpoint_model(
            args.checkpoint, DEVICE
        )

    # Determine if model is graph-based
    graph = architecture in ["SchNet", "EquiformerV2"]

    # Load dataset
    # Convenience for running all datasets
    if args.datasets[0] == "all":
        args.datasets = VALID_DATASETS

    # Download datasets if not present
    dataset_paths = []
    for dataset_name in args.datasets:
        dataset_path = Path(
            f"{args.datasets_base_path}/{args.split_name}/{dataset_name}"
        )
        if not dataset_path.exists():
            download_dataset(dataset_name, args.split_name, args.datasets_base_path)
        dataset_paths.append(dataset_path)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        dataset_paths,
        train_data_fraction=args.data_fraction if args.split_name == "train" else 0.0,
        batch_size=args.batch_size,
        seed=1024,
        architecture=architecture,
        batch_padded=False,
        val_data_fraction=args.data_fraction if args.split_name == "val" else 0.0,
        graph=graph,
    )

    # Select the appropriate loader
    loader = train_loader if args.split_name == "train" else val_loader

    # Compute metrics using fairchem's Evaluator
    metrics = compute_metrics(
        loader=loader, model=model, device=DEVICE, graph=graph, factorize=False
    )

    # Print metrics
    print("\nInference Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    # Determine model name for saving results
    if args.fairchem_model:
        model_name = f"FAIREquiformerV2_{args.fairchem_model}"
    else:
        model_name = Path(args.checkpoint).name.replace(".pth", "")

    # Save metrics to file
    with open(
        output_dir
        / f"{model_name}_eval_{args.split_name}_df={args.data_fraction}.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    if args.num_samples > 0:
        # Select the correct dataset based on split_name
        dataset = (
            train_loader.dataset if args.split_name == "train" else val_loader.dataset
        )

        # Use the existing collect_samples_helper to get raw samples
        raw_samples = collect_samples_helper(
            args.num_samples, dataset, model, graph, DEVICE
        )

        # Transform samples to match the visualization script's expected format
        formatted_samples = []
        formatted_predictions = []

        for sample_data in raw_samples:
            # Format ground truth sample for visualization
            formatted_sample = {
                "positions": sample_data["positions"],
                "atomic_numbers": sample_data["atomic_numbers"],
                "forces": sample_data["true"]["forces"],
                "energy": sample_data["true"]["energy"],
                "stress": sample_data["true"]["stress"],
            }
            formatted_samples.append(formatted_sample)

            # Format model predictions for visualization
            formatted_prediction = {
                "forces": sample_data["pred"]["forces"],
                "energy": sample_data["pred"]["energy"],
                "stress": sample_data["pred"]["stress"],
            }
            formatted_predictions.append(formatted_prediction)

        # Format the data in the structure expected by the visualization code
        samples_json = {
            -1: [
                {
                    "model_name": model_name,
                    "config": {
                        "num_params": param_count,
                        "dataset_size": -1,
                        "architecture": architecture,
                        "learning_rate": args.learning_rate,
                    },
                    "samples": {
                        args.split_name: formatted_samples,  # Ground truth data
                    },
                    "losses": {
                        "999": {  # Use a large epoch number for inference
                            "pred": {
                                args.split_name: formatted_predictions,  # Model predictions
                            }
                        }
                    },
                }
            ]
        }

        # Save structured sample data for visualization
        with open(output_dir / f"{model_name}_visualization_data.json", "w") as f:
            json.dump(samples_json, f, indent=4)

    print(f"\nInference completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
