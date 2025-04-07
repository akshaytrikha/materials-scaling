"""
Inference script for Transformer models, with integrated Evaluator for s2efs tasks
"""

# External
import torch
import ase
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from typing import Hashable

# Internal
from models.transformer_models import XTransformerModel
from data_utils import DATASET_INFO, custom_collate_fn_dataset_padded
from matrix import compute_distance_matrix, factorize_matrix
from data import OMat24Dataset

# Import the Evaluator and metric functions 
from fairchem.core.modules.evaluator import (
    Evaluator, metrics_dict, 
    mae, mse, cosine_similarity, magnitude_error,
    forcesx_mae, forcesy_mae, forcesz_mae,
    per_atom_mae
)

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)
# Ensure we're using CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    print("CUDA not available. Using CPU.")


# Define our new stress metric function
@metrics_dict
def stress_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> torch.Tensor:
    """Calculate whether stress predictions are within threshold."""
    s_thresh = 0.02  # 20 meV/Å³ threshold after conversion from eV
    error_stress = torch.abs(target["stress"] - prediction["stress"])
    max_errors = torch.max(error_stress, dim=1)[0]
    success = (max_errors < s_thresh).float()
    return success


class ExtendedEvaluator(Evaluator):
    """Extended evaluator that adds support for stress calculations."""
    
    def __init__(self, task="s2efs", eval_metrics=None):
        # Define stress metrics if not provided
        if eval_metrics is None and task == "s2efs":
            # Extend the s2ef task metrics to include stress
            eval_metrics = {
                "energy": ["mae", 
                           "per_atom_mae",
                           # "mse", 
                           # "energy_within_threshold"
                          ],
                "forces": [
                    # "forcesx_mae",
                    # "forcesy_mae",
                    # "forcesz_mae",
                    "mae",
                    "cosine_similarity",
                    # "magnitude_error",
                    # "energy_forces_within_threshold",
                ],
                "stress": [
                    "mae",
                    # "mse",
                    # "stress_within_threshold",
                ]
            }
        
        # Initialize the parent Evaluator with our extended metrics
        super().__init__(task=task, eval_metrics=eval_metrics)


def get_args():
    parser = argparse.ArgumentParser(description="Inference for trained Transformer models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["all"], 
        help="Datasets to run inference on, use 'all' to run on all valid datasets"
    )
    parser.add_argument(
        "--datasets_base_path", type=str, default="/workspace/akshay/omat24/datasets", 
        help="Base path for datasets"
    )
    parser.add_argument(
        "--split_name", type=str, default="val", 
        help="Split name (always val for inference)"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.json", 
        help="Path to save predictions"
    )
    parser.add_argument(
        "--n_elements", type=int, default=119, 
        help="Number of elements in vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--data_fractions", type=float, nargs="+", default=[1.0],
        help="Fraction of the dataset to use"
    )
    parser.add_argument(
        "--factorize", action="store_true", 
        help="Whether to use factorized distances"
    )
    parser.add_argument(
        "--sample_limit", type=int, default=None, 
        help="Limit the number of samples to process"
    )
    parser.add_argument(
        "--zero_baseline", action="store_true",
        help="Calculate baseline metrics using zero-force predictions"
    )
    # Model config parameters (optional, for loading models without saved configs)
    parser.add_argument(
        "--d_model", type=int, default=None, 
        help="Embedding dimension for transformer model (if not in checkpoint)"
    )
    parser.add_argument(
        "--depth", type=int, default=None, 
        help="Number of transformer layers (if not in checkpoint)"
    )
    parser.add_argument(
        "--n_heads", type=int, default=None, 
        help="Number of attention heads (if not in checkpoint)"
    )
    parser.add_argument(
        "--d_ff_mult", type=int, default=4, 
        help="Feed-forward network multiplier (if not in checkpoint)"
    )
    parser.add_argument(
        "--determine_config", action="store_true",
        help="Try to determine the model config from parameter count (if not in checkpoint)"
    )
    return parser.parse_args()


def determine_model_config(num_params):
    configurations = [
        {"d_model": 1, "depth": 1, "n_heads": 1, "d_ff_mult": 4, "params": 1778},
        {"d_model": 8, "depth": 2, "n_heads": 1, "d_ff_mult": 4, "params": 9657},
        {"d_model": 48, "depth": 3, "n_heads": 1, "d_ff_mult": 4, "params": 119758},
        {"d_model": 160, "depth": 3, "n_heads": 2, "d_ff_mult": 4, "params": 1019086},
        {"d_model": 384, "depth": 3, "n_heads": 2, "d_ff_mult": 4, "params": 4846798}
    ]
    
    # Find the closest match
    closest_config = min(configurations, key=lambda config: abs(config["params"] - num_params))
    
    return closest_config["d_model"], closest_config["depth"], closest_config["n_heads"], closest_config["d_ff_mult"]


def load_transformer_model(checkpoint_path, n_elements, split_name="val", use_factorized=False, 
                          manual_config=None, determine_config=False):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Check if model configuration is saved in the checkpoint
    if "model_config" in checkpoint:
        # Use the saved configuration
        config = checkpoint["model_config"]
        d_model = config.get("d_model")
        depth = config.get("depth")
        n_heads = config.get("n_heads")
        d_ff_mult = config.get("d_ff_mult", 4)
        use_factorized = config.get("use_factorized", use_factorized)
        print(f"Using saved model configuration: d_model={d_model}, depth={depth}, n_heads={n_heads}")
    
    # If no saved config, try manual config from user
    elif manual_config is not None and all(k in manual_config for k in ["d_model", "depth", "n_heads"]):
        d_model = manual_config["d_model"]
        depth = manual_config["depth"]
        n_heads = manual_config["n_heads"]
        d_ff_mult = manual_config.get("d_ff_mult", 4)
        print(f"Using user-provided model configuration: d_model={d_model}, depth={depth}, n_heads={n_heads}")
    
    # If user asked us to determine from parameter count
    elif determine_config:
        print("No configuration found in checkpoint. Trying to infer from parameter count...")
        # Get parameter count from state dict
        state_dict = checkpoint["model_state_dict"]
        num_params = sum(p.numel() for p in state_dict.values())
        
        # Try to determine the configuration
        d_model, depth, n_heads, d_ff_mult = determine_model_config(num_params)
        print(f"Inferred model configuration: d_model={d_model}, depth={depth}, n_heads={n_heads} (from {num_params} parameters)")
    
    # If we still don't have a config, error out with helpful message
    else:
        raise ValueError(
            "No model configuration found in checkpoint and no manual configuration provided.\n"
            "Please either:\n"
            "1. Update your training code to save model configuration in checkpoints, or\n"
            "2. Provide the model configuration via command line arguments:\n"
            "   --d_model VALUE --depth VALUE --n_heads VALUE [--d_ff_mult VALUE]\n"
            "3. Use --determine_config to attempt automatic determination (may be inaccurate)"
        )
    
    # Get max_seq_len
    max_seq_len = DATASET_INFO[split_name]["all"]["max_n_atoms"]
    
    # Initialize the model
    model = XTransformerModel(
        num_tokens=n_elements,
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        d_ff_mult=d_ff_mult,
        use_factorized=use_factorized
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    return model


def create_dataloader(dataset_path, batch_size, use_factorized, num_workers, sample_limit=None):
    # Load the dataset
    dataset = OMat24Dataset(
        dataset_paths=[Path(dataset_path)], 
        architecture="Transformer",
        graph=False
    )
    
    # Limit samples if requested
    if sample_limit is not None:
        dataset_size = min(len(dataset), sample_limit)
        indices = list(range(dataset_size))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Get max_n_atoms from the dataset info
    split_name = dataset_path.parts[-2]  # Extract split name from path
    max_n_atoms = DATASET_INFO[split_name]["all"]["max_n_atoms"]
    
    # Create a collate function that pads to the dataset's max atoms
    collate_fn = lambda batch: custom_collate_fn_dataset_padded(batch, max_n_atoms, use_factorized)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    return dataloader, len(dataset)


def run_inference_batched(model, dataloader, use_factorized=False):
    """Run inference with batched data processing"""
    predictions = []
    
    # Process in batches
    for batch in tqdm(dataloader, desc="Processing batches"):
        with torch.no_grad():
            # Move data to device
            atomic_numbers = batch["atomic_numbers"].to(DEVICE, non_blocking=True)
            positions = batch["positions"].to(DEVICE, non_blocking=True)
            true_forces = batch["forces"].to(DEVICE, non_blocking=True)
            true_energy = batch["energy"].to(DEVICE, non_blocking=True)
            true_stress = batch["stress"].to(DEVICE, non_blocking=True)
            
            # Create mask for padded atoms
            mask = atomic_numbers != 0
            
            # Forward pass
            if use_factorized:
                factorized_matrix = batch["factorized_matrix"].to(DEVICE, non_blocking=True)
                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers, positions, factorized_matrix, mask
                )
            else:
                distance_matrix = batch["distance_matrix"].to(DEVICE, non_blocking=True)
                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers, positions, distance_matrix, mask
                )
            
            # Process each sample in the batch
            batch_size = atomic_numbers.size(0)
            for i in range(batch_size):
                # Get the number of actual atoms (non-padded)
                sample_mask = mask[i]
                sample_length = sample_mask.sum().item()
                
                # Extract predictions for this sample
                prediction = {
                    "idx": i,  # Note: this is batch index, not original dataset index
                    "atomic_numbers": atomic_numbers[i, :sample_length].cpu().tolist(),
                    "positions": positions[i, :sample_length].cpu().tolist(),
                    "true": {
                        "forces": true_forces[i, :sample_length].cpu().tolist(),
                        "energy": true_energy[i].cpu().item(),
                        "stress": true_stress[i].cpu().tolist(),
                    },
                    "pred": {
                        "forces": pred_forces[i, :sample_length].cpu().tolist(),
                        "energy": pred_energy[i].cpu().item(),
                        "stress": pred_stress[i].cpu().tolist(),
                    },
                }
                
                predictions.append(prediction)
    
    return predictions


def compute_metrics(predictions, calculate_zero_baseline=False):
    """Compute metrics using the Extended Evaluator."""
    # Format predictions to match the Evaluator's expected input format
    prediction_tensor = {"energy": [], "forces": [], "stress": []}
    target_tensor = {"energy": [], "forces": [], "stress": [], "natoms": []}
    
    # Convert predictions to the format expected by the Evaluator
    for pred in predictions:
        # Get number of atoms for this structure
        n_atoms = len(pred["atomic_numbers"])
        
        # Append energy (per structure)
        prediction_tensor["energy"].append(pred["pred"]["energy"])
        target_tensor["energy"].append(pred["true"]["energy"])
        
        # Append forces (per atom)
        prediction_tensor["forces"].extend(pred["pred"]["forces"])
        target_tensor["forces"].extend(pred["true"]["forces"])
        
        # Append stress (per structure)
        prediction_tensor["stress"].append(pred["pred"]["stress"])
        target_tensor["stress"].append(pred["true"]["stress"])
        
        # Keep track of number of atoms per structure
        target_tensor["natoms"].append(n_atoms)
    
    # Convert lists to tensors
    for key in prediction_tensor:
        prediction_tensor[key] = torch.tensor(prediction_tensor[key])
    
    for key in target_tensor:
        target_tensor[key] = torch.tensor(target_tensor[key])
    
    # Initialize our extended evaluator for s2efs task
    evaluator = ExtendedEvaluator(task="s2efs")
    
    # Evaluate the predictions
    all_metrics = evaluator.eval(prediction_tensor, target_tensor)
    
    # Extract the computed metrics
    metrics = {
        "energy_mae_meV": all_metrics["energy_mae"]["metric"] * 1000,  # Convert eV to meV
        "energy_mae_meV_per_atom": all_metrics["energy_per_atom_mae"]["metric"] * 1000,  # Use per-atom metric for fairer comparison
        "force_mae_meV_per_A": all_metrics["forces_mae"]["metric"] * 1000,      # Convert eV/Å to meV/Å
        "force_cos_similarity": all_metrics["forces_cosine_similarity"]["metric"],
        "stress_mae_meV_per_A3": all_metrics["stress_mae"]["metric"] * 1000,    # Convert eV/Å³ to meV/Å³
    }
    
    # Calculate zero-force baseline if requested
    if calculate_zero_baseline:
        zero_force_mae = 0.0
        total_force_components = 0
        
        for pred in predictions:
            true_forces = np.array(pred["true"]["forces"])
            zero_forces = np.zeros_like(true_forces)
            zero_force_errors = np.abs(true_forces).flatten() * 1000  # Convert to meV/Å
            zero_force_mae += np.sum(zero_force_errors)
            total_force_components += zero_force_errors.size
        
        metrics["zero_force_mae_meV_per_A"] = zero_force_mae / total_force_components if total_force_components > 0 else 0.0
    
    # Compute raw counts for weighted averaging
    raw_counts = {
        "n_samples": len(predictions),
        "n_atoms": all_metrics["energy_mae"]["numel"],
        "n_atoms_with_valid_forces": all_metrics["forces_cosine_similarity"]["numel"],
        "skipped_true_zero": 0,  # The Evaluator doesn't track this separately
        "skipped_pred_zero": 0,  # The Evaluator doesn't track this separately
        "force_components": all_metrics["forces_mae"]["numel"],
        "stress_components": all_metrics["stress_mae"]["numel"],
    }
    
    return metrics, raw_counts


def compute_weighted_average_metrics(all_dataset_results):
    weighted_metrics = {
        "energy_mae_meV": 0.0,
        "energy_mae_meV_per_atom": 0.0,
        "force_mae_meV_per_A": 0.0,
        "force_cos_similarity": 0.0,
        "stress_mae_meV_per_A3": 0.0,
    }
    
    # Check if we have zero-force baseline metrics
    if "zero_force_mae_meV_per_A" in next(iter(all_dataset_results.values()))["metrics"]:
        weighted_metrics["zero_force_mae_meV_per_A"] = 0.0
    
    # Collect totals for weighted averaging
    total_samples = 0
    total_atoms = 0
    total_atoms_with_valid_forces = 0
    total_skipped_true_zero = 0
    total_skipped_pred_zero = 0
    total_force_components = 0
    total_stress_components = 0
    
    # Sum up the weighted contributions
    for dataset_key, result in all_dataset_results.items():
        metrics = result["metrics"]
        raw_counts = result["raw_counts"]
        
        n_samples = raw_counts["n_samples"]
        n_atoms = raw_counts["n_atoms"]
        n_atoms_with_valid_forces = raw_counts["n_atoms_with_valid_forces"]
        skipped_true_zero = raw_counts["skipped_true_zero"]
        skipped_pred_zero = raw_counts["skipped_pred_zero"]
        n_force_components = raw_counts["force_components"]
        n_stress_components = raw_counts["stress_components"]
        
        # Accumulate totals for weighting
        total_samples += n_samples
        total_atoms += n_atoms
        total_atoms_with_valid_forces += n_atoms_with_valid_forces
        total_skipped_true_zero += skipped_true_zero
        total_skipped_pred_zero += skipped_pred_zero
        total_force_components += n_force_components
        total_stress_components += n_stress_components
        
        # Regular energy MAE is weighted by number of samples
        weighted_metrics["energy_mae_meV"] += metrics["energy_mae_meV"] * n_samples
        
        # Per-atom energy MAE is already normalized, so we can just weight by number of atoms
        weighted_metrics["energy_mae_meV_per_atom"] += metrics["energy_mae_meV_per_atom"] * n_atoms
        
        # Force errors are measured per component, so weight by number of force components
        weighted_metrics["force_mae_meV_per_A"] += metrics["force_mae_meV_per_A"] * n_force_components
        
        # Zero-force baseline if available
        if "zero_force_mae_meV_per_A" in metrics:
            weighted_metrics["zero_force_mae_meV_per_A"] += metrics["zero_force_mae_meV_per_A"] * n_force_components
        
        # Force cosine similarity is per atom, so weight by number of atoms with valid forces
        weighted_metrics["force_cos_similarity"] += metrics["force_cos_similarity"] * n_atoms_with_valid_forces
        
        # Stress errors are measured per component, so weight by number of stress components
        weighted_metrics["stress_mae_meV_per_A3"] += metrics["stress_mae_meV_per_A3"] * n_stress_components
    
    # Normalize by the totals
    weighted_metrics["energy_mae_meV"] /= total_samples if total_samples > 0 else 1
    weighted_metrics["energy_mae_meV_per_atom"] /= total_atoms if total_atoms > 0 else 1
    weighted_metrics["force_mae_meV_per_A"] /= total_force_components if total_force_components > 0 else 1
    if "zero_force_mae_meV_per_A" in weighted_metrics:
        weighted_metrics["zero_force_mae_meV_per_A"] /= total_force_components if total_force_components > 0 else 1
    weighted_metrics["force_cos_similarity"] /= total_atoms_with_valid_forces if total_atoms_with_valid_forces > 0 else 1
    weighted_metrics["stress_mae_meV_per_A3"] /= total_stress_components if total_stress_components > 0 else 1
    
    # Add count information to the return value
    counts_info = {
        "total_samples": total_samples,
        "total_atoms": total_atoms,
        "total_atoms_with_valid_forces": total_atoms_with_valid_forces,
        "total_skipped_true_zero": total_skipped_true_zero,
        "total_skipped_pred_zero": total_skipped_pred_zero,
        "total_force_components": total_force_components,
        "total_stress_components": total_stress_components,
    }
    
    return weighted_metrics, counts_info


def main():
    args = get_args()
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    
    # Prepare manual config if provided
    manual_config = None
    if args.d_model is not None and args.depth is not None and args.n_heads is not None:
        manual_config = {
            "d_model": args.d_model,
            "depth": args.depth,
            "n_heads": args.n_heads,
            "d_ff_mult": args.d_ff_mult
        }
    
    # Load the model
    try:
        model = load_transformer_model(
            args.checkpoint, 
            args.n_elements, 
            args.split_name, 
            args.factorize,
            manual_config=manual_config,
            determine_config=args.determine_config
        )
    except ValueError as e:
        print(f"Error loading model: {e}")
        return
    
    # Handle 'all' datasets option
    from data_utils import VALID_DATASETS
    if args.datasets[0].lower() == "all":
        args.datasets = VALID_DATASETS
        print(f"Running on all valid datasets: {', '.join(args.datasets)}")
    
    # Track all results
    all_results = {}
    
    # Process each dataset and data fraction combination
    for dataset_name in args.datasets:
        dataset_path = Path(f"{args.datasets_base_path}/{args.split_name}/{dataset_name}")
        
        # Make sure dataset exists
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            from data_utils import download_dataset, VALID_DATASETS
            if dataset_name in VALID_DATASETS:
                print(f"Downloading dataset: {dataset_name}")
                download_dataset(dataset_name, args.split_name, args.datasets_base_path)
            else:
                print(f"Invalid dataset name: {dataset_name}")
                continue
        
        for fraction in args.data_fractions:
            print(f"Running inference on dataset: {dataset_name} with fraction: {fraction}")
            
            # Calculate the sample limit based on the fraction
            sample_limit = None
            if fraction < 1.0:
                dataset_size = len(OMat24Dataset([dataset_path], "Transformer"))
                sample_limit = int(dataset_size * fraction)
            
            # Create the dataloader with batch processing
            dataloader, dataset_size = create_dataloader(
                dataset_path, 
                args.batch_size, 
                args.factorize,
                args.num_workers,
                sample_limit
            )
            
            print(f"Created dataloader with {dataset_size} samples, batch size {args.batch_size}, {args.num_workers} workers")
            
            # Run inference
            predictions = run_inference_batched(
                model, 
                dataloader, 
                args.factorize
            )
            
            # Compute metrics (including zero-force baseline if requested)
            metrics, raw_counts = compute_metrics(predictions, args.zero_baseline)
            print(f"Evaluation metrics for {dataset_name} (fraction {fraction}):")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
                # Print relative improvement over zero baseline if available
                if args.zero_baseline and metric_name == "force_mae_meV_per_A" and "zero_force_mae_meV_per_A" in metrics:
                    zero_mae = metrics["zero_force_mae_meV_per_A"]
                    improvement = (zero_mae - value) / zero_mae * 100 if zero_mae > 0 else 0
                    print(f"  Force MAE improvement over zero baseline: {improvement:.2f}%")
            
            # Print the counts of skipped samples
            print(f"  Skipped atoms (true force=0): {raw_counts['skipped_true_zero']}")
            print(f"  Skipped atoms (pred force=0): {raw_counts['skipped_pred_zero']}")
            
            # Store results
            result_key = f"{dataset_name}_{fraction}"
            all_results[result_key] = {
                "metrics": metrics,
                "raw_counts": raw_counts
            }
    
    # Calculate aggregate metrics across all datasets using weighted averages
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("WEIGHTED AVERAGE METRICS ACROSS ALL DATASETS:")
        print("="*80)
        
        # Calculate weighted averages across all datasets
        weighted_metrics, count_info = compute_weighted_average_metrics(all_results)
        
        print(f"Combined results from {count_info['total_samples']} samples with {count_info['total_atoms']} atoms:")
        for metric_name, value in weighted_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
            # Print relative improvement over zero baseline if available
            if args.zero_baseline and metric_name == "force_mae_meV_per_A" and "zero_force_mae_meV_per_A" in weighted_metrics:
                zero_mae = weighted_metrics["zero_force_mae_meV_per_A"]
                improvement = (zero_mae - value) / zero_mae * 100 if zero_mae > 0 else 0
                print(f"  Force MAE improvement over zero baseline: {improvement:.2f}%")
        
        # Print the total counts of skipped samples
        print(f"  Total atoms used for force cosine similarity: {count_info['total_atoms_with_valid_forces']}")
        print(f"  Total skipped atoms (true force=0): {count_info['total_skipped_true_zero']}")
        print(f"  Total skipped atoms (pred force=0): {count_info['total_skipped_pred_zero']}")
    
    # Save metrics only (not individual predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create a structure with weighted averages
    weighted_results, count_info = compute_weighted_average_metrics(all_results)
    
    # Extract just the metrics from each dataset result (without the predictions)
    dataset_metrics = {}
    for dataset_key, result in all_results.items():
        dataset_metrics[dataset_key] = {
            "metrics": result["metrics"],
            "raw_counts": result["raw_counts"]
        }
    
    output_data = {
        "dataset_metrics": dataset_metrics,  # Only metrics per dataset, not predictions
        "weighted_metrics": weighted_results,
        "count_info": count_info,
        "config": {
            "checkpoint": args.checkpoint,
            "datasets": args.datasets,
            "split_name": args.split_name,
            "n_elements": args.n_elements,
            "data_fractions": args.data_fractions,
            "factorize": args.factorize,
            "zero_baseline": args.zero_baseline,
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()