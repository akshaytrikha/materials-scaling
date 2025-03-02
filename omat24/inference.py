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

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch_geometric.deprecation"
)

# Internal
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS
from train_utils import forward_pass, collect_samples_for_visualizing
from models.fcn import FCNModel
from models.transformer_models import XTransformerModel
from models.schnet import SchNet
from models.equiformer_v2 import EquiformerS2EF

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
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

    return model, architecture


def compute_metrics(loader, model, device, graph=False, factorize=False):
    """Compute evaluation metrics in a batched manner without loading the entire dataset.

    Args:
        loader: DataLoader for the evaluation data
        model: The model to evaluate
        device: The device to run evaluation on
        graph: Whether using graph-based models
        factorize: Whether using factorized distances

    Returns:
        dict: Metrics dictionary with values in meV units, matching the paper's metrics
    """
    model.eval()

    # Initialize accumulators for metrics
    sum_force_abs_error = 0.0
    sum_force_sq_error = 0.0
    sum_energy_abs_error_per_atom = 0.0
    sum_energy_sq_error_per_atom = 0.0
    sum_stress_abs_error = 0.0
    sum_stress_sq_error = 0.0

    # For R² calculation
    sum_energy = 0.0
    sum_energy_squared = 0.0
    sum_energy_pred_error_squared = 0.0

    # For force cosine similarity
    sum_force_cos_sim = 0.0
    count_force_vectors = 0

    # Total counts for averaging
    total_atoms = 0
    total_structures = 0
    total_stress_components = 0

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

            # Convert to meV (multiply by 1000)
            pred_forces_meV = pred_forces * 1000
            true_forces_meV = true_forces * 1000
            pred_energy_meV = pred_energy * 1000
            true_energy_meV = true_energy * 1000
            pred_stress_meV = pred_stress * 1000
            true_stress_meV = true_stress * 1000

            # Get batch size
            batch_size = pred_energy.shape[0]
            total_structures += batch_size

            # Force metrics
            if mask is not None and not graph:
                # Apply mask to forces for non-graph models
                mask = mask.unsqueeze(-1).expand_as(pred_forces)
                pred_forces_flat = pred_forces_meV.masked_select(mask).reshape(-1, 3)
                true_forces_flat = true_forces_meV.masked_select(mask).reshape(-1, 3)
            elif len(pred_forces_meV.shape) == 3:  # batch x atoms x 3
                # Reshape to [total_atoms, 3]
                pred_forces_flat = pred_forces_meV.reshape(-1, 3)
                true_forces_flat = true_forces_meV.reshape(-1, 3)
            else:  # Already [N_atoms, 3]
                pred_forces_flat = pred_forces_meV
                true_forces_flat = true_forces_meV

            # Count total atoms in this batch for forces
            batch_atoms = pred_forces_flat.shape[0]
            total_atoms += batch_atoms

            # Force MAE and MSE
            force_abs_error = (
                torch.abs(pred_forces_flat - true_forces_flat).sum().item()
            )
            force_sq_error = ((pred_forces_flat - true_forces_flat) ** 2).sum().item()

            sum_force_abs_error += force_abs_error
            sum_force_sq_error += force_sq_error

            # Force cosine similarity
            for i in range(pred_forces_flat.shape[0]):
                true_norm = torch.norm(true_forces_flat[i])
                pred_norm = torch.norm(pred_forces_flat[i])

                # Skip atoms with near-zero force
                if true_norm > 1e-6 and pred_norm > 1e-6:
                    cos_sim = torch.dot(true_forces_flat[i], pred_forces_flat[i]) / (
                        true_norm * pred_norm
                    )
                    sum_force_cos_sim += cos_sim.item()
                    count_force_vectors += 1

            # Energy per-atom metrics
            energy_abs_err = torch.abs(pred_energy_meV - true_energy_meV)
            energy_sq_err = (pred_energy_meV - true_energy_meV) ** 2

            # Normalize by number of atoms in each structure
            energy_abs_err_per_atom = energy_abs_err / natoms
            energy_sq_err_per_atom = energy_sq_err / natoms

            sum_energy_abs_error_per_atom += energy_abs_err_per_atom.sum().item()
            sum_energy_sq_error_per_atom += energy_sq_err_per_atom.sum().item()

            # For R² calculation
            sum_energy += true_energy.sum().item()
            sum_energy_squared += (true_energy**2).sum().item()
            sum_energy_pred_error_squared += (
                ((true_energy - pred_energy) ** 2).sum().item()
            )

            # Stress metrics
            # Count total stress components
            batch_stress_components = pred_stress.numel()
            total_stress_components += batch_stress_components

            stress_abs_error = torch.abs(pred_stress_meV - true_stress_meV).sum().item()
            stress_sq_error = ((pred_stress_meV - true_stress_meV) ** 2).sum().item()

            sum_stress_abs_error += stress_abs_error
            sum_stress_sq_error += stress_sq_error

    # Calculate final metrics
    # Force metrics
    force_mae = sum_force_abs_error / (
        total_atoms * 3
    )  # Divide by total vector components
    force_rmse = np.sqrt(sum_force_sq_error / (total_atoms * 3))
    force_cos = (
        sum_force_cos_sim / count_force_vectors if count_force_vectors > 0 else 0
    )

    # Energy metrics
    energy_mae = sum_energy_abs_error_per_atom / total_structures
    energy_rmse = np.sqrt(sum_energy_sq_error_per_atom / total_structures)

    # Calculate R²
    energy_mean = sum_energy / total_structures
    energy_ss_tot = sum_energy_squared - total_structures * (energy_mean**2)
    energy_r2 = 1 - (sum_energy_pred_error_squared / energy_ss_tot)

    # Stress metrics
    stress_mae = sum_stress_abs_error / total_stress_components
    stress_rmse = np.sqrt(sum_stress_sq_error / total_stress_components)

    return {
        "force_mae": force_mae,  # meV/Å
        "energy_mae": energy_mae,  # meV/atom
        "stress_mae": stress_mae,  # meV/Å³
        "force_rmse": force_rmse,  # meV/Å
        "energy_rmse": energy_rmse,  # meV/atom
        "stress_rmse": stress_rmse,  # meV/Å³
        "energy_r2": energy_r2,  # dimensionless
        "force_cos": force_cos,  # dimensionless (higher is better)
    }


def main():
    # Parse arguments
    args = parse_args()
    output_dir = Path("inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    # elif args.architecture == "EquiformerV2":
    #     if DEVICE == torch.device("mps"):
    #         print("MPS is not supported for EquiformerV2. Switching to CPU.")
    #         DEVICE = torch.device("cpu")
    # print(f"Using device: {DEVICE}")

    # Load model - either from checkpoint or FAIRChem
    if args.fairchem_model:
        # Load FAIRChem model
        model, architecture = load_fairchem_eqV2(args.fairchem_model, DEVICE)
    else:
        # Load model from checkpoint
        model, architecture = load_checkpoint_model(args.checkpoint, DEVICE)

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

    # Compute metrics in a batched manner
    metrics = compute_metrics(
        loader=loader, model=model, device=DEVICE, graph=graph, factorize=False
    )

    # Print metrics
    print("\nInference Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    # Determine model name for saving results
    if args.fairchem_model:
        model_name = f"FAIREquiformerV2"
    else:
        model_name = Path(args.checkpoint).name.replace(".pth", "")

    # Save metrics to file
    with open(
        output_dir
        / f"{model_name}_eval_{args.split_name}_df={args.data_fraction}.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    # Optional: collect sample visualizations if needed
    if args.num_samples > 0:
        # Use the collect_samples_for_visualizing function from train_utils.py
        samples = collect_samples_for_visualizing(
            model,
            graph,
            train_loader,
            val_loader,
            DEVICE,
            args.num_samples,
        )

        # Save sample data for further analysis
        with open(output_dir / f"{model_name}_samples.json", "w") as f:
            json.dump(samples, f, indent=4)

        print(
            f"Saved {args.num_samples} visualization samples to {output_dir / f'{model_name}_samples.json'}"
        )

    print(f"\nInference completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
