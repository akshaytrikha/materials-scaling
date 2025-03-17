# External
import torch
import ase
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Internal
from models.transformer_models import XTransformerModel
from data_utils import DATASET_INFO
from matrix import compute_distance_matrix, factorize_matrix
from data import OMat24Dataset

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


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference for trained Transformer models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["all"], 
        help="Datasets to run inference on, use 'all' to run on all valid datasets"
    )
    parser.add_argument(
        "--datasets_base_path", type=str, default="./datasets", 
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
    return parser.parse_args()


def load_transformer_model(checkpoint_path, n_elements, split_name="val", use_factorized=False):
    """Load a trained transformer model from a checkpoint file."""
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Check if model configuration is saved in the checkpoint
    if "model_config" in checkpoint:
        # Use the saved configuration
        config = checkpoint["model_config"]
        d_model = config.get("d_model")
        depth = config.get("depth")
        n_heads = config.get("n_heads")
        d_ff_mult = config.get("d_ff_mult") 
        use_factorized = config.get("use_factorized", use_factorized)
        print(f"Using saved model configuration: d_model={d_model}, depth={depth}, n_heads={n_heads}")
    else:
        # Error out if no configuration is found
        raise ValueError(
            "No model configuration found in checkpoint. Please update your training code to "
            "save model configuration in checkpoints using the provided snippet."
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


def run_inference(model, dataset_path, batch_size=32, use_factorized=False, sample_limit=None):
    """Run inference on a dataset and return predictions."""
    # Load the dataset
    dataset = OMat24Dataset(
        dataset_paths=[Path(dataset_path)], 
        architecture="Transformer",
        graph=False
    )
    
    # Limit samples if requested
    if sample_limit is not None:
        dataset_size = min(len(dataset), sample_limit)
    else:
        dataset_size = len(dataset)
    
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, dataset_size, batch_size), desc="Processing batches"):
        batch_indices = range(i, min(i + batch_size, dataset_size))
        batch_samples = [dataset[j] for j in batch_indices]
        
        # Process each sample in the batch
        for sample in batch_samples:
            with torch.no_grad():
                # Extract data
                atomic_numbers = sample["atomic_numbers"].unsqueeze(0).to(DEVICE)
                positions = sample["positions"].unsqueeze(0).to(DEVICE)
                true_forces = sample["forces"].unsqueeze(0).to(DEVICE)
                true_energy = sample["energy"].unsqueeze(0).to(DEVICE)
                true_stress = sample["stress"].unsqueeze(0).to(DEVICE)
                symbols = sample["symbols"]
                idx = sample["idx"]
                
                # Create mask for padded atoms
                mask = atomic_numbers != 0
                
                # Forward pass
                if use_factorized:
                    factorized_distances = sample["factorized_matrix"].unsqueeze(0).to(DEVICE)
                    pred_forces, pred_energy, pred_stress = model(
                        atomic_numbers, positions, factorized_distances, mask
                    )
                else:
                    distance_matrix = sample["distance_matrix"].unsqueeze(0).to(DEVICE)
                    pred_forces, pred_energy, pred_stress = model(
                        atomic_numbers, positions, distance_matrix, mask
                    )
                
                # Get the number of actual atoms (non-padded)
                sample_length = mask.sum(dim=1)[0].item()
                
                # Extract predictions
                prediction = {
                    "idx": idx,
                    "symbols": symbols,
                    "atomic_numbers": atomic_numbers[0, :sample_length].cpu().tolist(),
                    "positions": positions[0, :sample_length].cpu().tolist(),
                    "true": {
                        "forces": true_forces[0, :sample_length].cpu().tolist(),
                        "energy": true_energy.cpu().item(),
                        "stress": true_stress.cpu().tolist(),
                    },
                    "pred": {
                        "forces": pred_forces[0, :sample_length].cpu().tolist(),
                        "energy": pred_energy.cpu().item(),
                        "stress": pred_stress[0].cpu().tolist(),
                    },
                }
                
                predictions.append(prediction)
    
    return predictions


def compute_metrics(predictions):
    """
    Compute evaluation metrics from predictions.
    
    Energy errors are in units meV/atom
    Forces errors are in meV/Å
    Stress errors are in meV/Å^3
    
    The original dataset contains structures labeled with:
    - total energy (eV)
    - forces (eV/A)
    - stress (eV/A^3)
    """
    metrics = {
        "energy_mae_meV_per_atom": 0.0,
        "force_mae_meV_per_A": 0.0,
        "stress_mae_meV_per_A3": 0.0,
        "stress_iso_mae_meV_per_A3": 0.0,
        "stress_aniso_mae_meV_per_A3": 0.0,
    }
    
    # Conversion factor from eV to meV
    eV_to_meV = 1000.0
    
    # For energy per atom calculation
    total_atoms = 0
    total_energy_error = 0.0
    
    # For calculating MAE across all force components
    all_force_errors = []
    
    # For stress MAE calculation
    all_stress_errors = []
    all_stress_iso_errors = []
    all_stress_aniso_errors = []
    
    for pred in predictions:
        # Get number of atoms in this structure
        n_atoms = len(pred["atomic_numbers"])
        total_atoms += n_atoms
        
        # Energy error (to be divided by total atoms later)
        energy_error = abs(pred["pred"]["energy"] - pred["true"]["energy"]) * eV_to_meV
        total_energy_error += energy_error
        
        # Force errors in meV/Å
        true_forces = np.array(pred["true"]["forces"])
        pred_forces = np.array(pred["pred"]["forces"])
        
        # Calculate element-wise absolute errors for all force components
        force_errors = np.abs(true_forces - pred_forces).flatten() * eV_to_meV
        all_force_errors.extend(force_errors.tolist())
        
        # Stress errors in meV/Å^3
        true_stress = np.array(pred["true"]["stress"])
        pred_stress = np.array(pred["pred"]["stress"])
        
        # Calculate element-wise absolute errors for all stress components
        stress_errors = np.abs(true_stress - pred_stress) * eV_to_meV
        all_stress_errors.extend(stress_errors.tolist())
        
        # Separate isotropic and anisotropic stress components
        true_p = true_stress[:3].mean()
        true_iso_stress = np.array([true_p, true_p, true_p, 0, 0, 0])
        true_aniso_stress = true_stress - true_iso_stress
        
        pred_p = pred_stress[:3].mean()
        pred_iso_stress = np.array([pred_p, pred_p, pred_p, 0, 0, 0])
        pred_aniso_stress = pred_stress - pred_iso_stress
        
        # Calculate element-wise absolute errors for isotropic and anisotropic components
        iso_stress_errors = np.abs(true_iso_stress - pred_iso_stress) * eV_to_meV
        aniso_stress_errors = np.abs(true_aniso_stress - pred_aniso_stress) * eV_to_meV
        
        all_stress_iso_errors.extend(iso_stress_errors.tolist())
        all_stress_aniso_errors.extend(aniso_stress_errors.tolist())
    
    # Calculate final metrics
    # Energy: meV/atom (divide total energy error by total number of atoms)
    metrics["energy_mae_meV_per_atom"] = total_energy_error / total_atoms
    
    # Forces: meV/Å (average of all force component errors)
    metrics["force_mae_meV_per_A"] = np.mean(all_force_errors)
    
    # Stress: meV/Å^3 (average of all stress component errors)
    metrics["stress_mae_meV_per_A3"] = np.mean(all_stress_errors)
    metrics["stress_iso_mae_meV_per_A3"] = np.mean(all_stress_iso_errors)
    metrics["stress_aniso_mae_meV_per_A3"] = np.mean(all_stress_aniso_errors)
    
    # Return metrics along with the raw data needed for weighted averaging
    raw_counts = {
        "n_samples": len(predictions),
        "n_atoms": total_atoms,
        "force_components": len(all_force_errors),
        "stress_components": len(all_stress_errors),
    }
    
    return metrics, raw_counts


def compute_weighted_average_metrics(all_dataset_results):
    """
    Compute weighted average metrics across all datasets, where weights are determined
    by the appropriate counts (atoms for energy, force components for forces, etc.)
    
    Args:
        all_dataset_results: Dictionary of {dataset_name: {predictions, metrics, raw_counts}}
        
    Returns:
        Dictionary of weighted average metrics
    """
    weighted_metrics = {
        "energy_mae_meV_per_atom": 0.0,
        "force_mae_meV_per_A": 0.0,
        "stress_mae_meV_per_A3": 0.0,
        "stress_iso_mae_meV_per_A3": 0.0,
        "stress_aniso_mae_meV_per_A3": 0.0,
    }
    
    # Collect totals for weighted averaging
    total_samples = 0
    total_atoms = 0
    total_force_components = 0
    total_stress_components = 0
    
    # Sum up the weighted contributions
    for dataset_key, result in all_dataset_results.items():
        metrics = result["metrics"]
        raw_counts = result["raw_counts"]
        
        n_samples = raw_counts["n_samples"]
        n_atoms = raw_counts["n_atoms"]
        n_force_components = raw_counts["force_components"]
        n_stress_components = raw_counts["stress_components"]
        
        # Accumulate totals for weighting
        total_samples += n_samples
        total_atoms += n_atoms
        total_force_components += n_force_components
        total_stress_components += n_stress_components
        
        # Energy error is already per-atom, so weight by number of atoms
        weighted_metrics["energy_mae_meV_per_atom"] += metrics["energy_mae_meV_per_atom"] * n_atoms
        
        # Force errors are measured per component, so weight by number of force components
        weighted_metrics["force_mae_meV_per_A"] += metrics["force_mae_meV_per_A"] * n_force_components
        
        # Stress errors are measured per component, so weight by number of stress components
        weighted_metrics["stress_mae_meV_per_A3"] += metrics["stress_mae_meV_per_A3"] * n_stress_components
        weighted_metrics["stress_iso_mae_meV_per_A3"] += metrics["stress_iso_mae_meV_per_A3"] * n_stress_components
        weighted_metrics["stress_aniso_mae_meV_per_A3"] += metrics["stress_aniso_mae_meV_per_A3"] * n_stress_components
    
    # Normalize by the totals
    weighted_metrics["energy_mae_meV_per_atom"] /= total_atoms
    weighted_metrics["force_mae_meV_per_A"] /= total_force_components
    weighted_metrics["stress_mae_meV_per_A3"] /= total_stress_components
    weighted_metrics["stress_iso_mae_meV_per_A3"] /= total_stress_components
    weighted_metrics["stress_aniso_mae_meV_per_A3"] /= total_stress_components
    
    # Add count information to the return value
    counts_info = {
        "total_samples": total_samples,
        "total_atoms": total_atoms,
        "total_force_components": total_force_components,
        "total_stress_components": total_stress_components,
    }
    
    return weighted_metrics, counts_info


def main():
    args = get_args()
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    # Check if the checkpoint is a transformer model
    if "Transformer" not in args.checkpoint:
        print("Warning: The checkpoint may not be a Transformer model. Inferring from parameter count.")
    
    # Load the model
    model = load_transformer_model(
        args.checkpoint, 
        args.n_elements, 
        args.split_name, 
        args.factorize
    )
    
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
            
            # Run inference
            predictions = run_inference(
                model, 
                dataset_path, 
                args.batch_size, 
                args.factorize,
                int(len(OMat24Dataset([dataset_path], "Transformer")) * fraction) if args.sample_limit is None else args.sample_limit
            )
            
            # Compute metrics
            metrics, raw_counts = compute_metrics(predictions)
            print(f"Evaluation metrics for {dataset_name} (fraction {fraction}):")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            # Store results
            result_key = f"{dataset_name}_{fraction}"
            all_results[result_key] = {
                "predictions": predictions,
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
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create a structure with weighted averages
    weighted_results, count_info = compute_weighted_average_metrics(all_results)
    
    output_data = {
        "results": all_results,
        "weighted_metrics": weighted_results,
        "count_info": count_info,
        "config": {
            "checkpoint": args.checkpoint,
            "datasets": args.datasets,
            "split_name": args.split_name,
            "n_elements": args.n_elements,
            "data_fractions": args.data_fractions,
            "factorize": args.factorize,
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()