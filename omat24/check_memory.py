import argparse
import time
import torch
import json
import gc
from pathlib import Path

from models.fcn import MetaFCNModels
from models.transformer_models import MetaTransformerModels
from models.schnet import MetaSchNetModels
from models.equiformer_v2 import MetaEquiformerV2Models
from models.adit import MetaADiTModels
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS, DATASET_INFO
from train_utils import forward_pass


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check memory usage of models during forward passes."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["FCN", "Transformer", "SchNet", "EquiformerV2", "ADiT"],
        default="SchNet",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for forward pass"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rattled-300-subsampled",
        choices=VALID_DATASETS,
        help="Dataset to use for testing",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="val",
        help="OMat24 split to use",
    )
    parser.add_argument(
        "--datasets_base_path",
        type=str,
        default="./datasets",
        help="Base path for dataset storage",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of forward pass iterations to run",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="memory_stats.json",
        help="Output file for memory statistics",
    )

    return parser.parse_args()


def log_memory_stats(device):
    """Log current memory usage statistics."""
    if device.type == "cuda":
        stats = {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
    else:
        stats = {"device": str(device), "memory_stats_not_available": True}
    
    return stats


def init_meta_models(architecture, device):
    """Initialize meta model class based on architecture."""
    print(f"Initializing {architecture} meta models...")

    # Initialize meta model class based on architecture choice
    if architecture == "FCN":
        meta_models = MetaFCNModels(vocab_size=119)
    elif architecture == "Transformer":
        meta_models = MetaTransformerModels(
            vocab_size=119,
            max_seq_len=DATASET_INFO["val"]["all"]["max_n_atoms"],
        )
    elif architecture == "SchNet":
        meta_models = MetaSchNetModels(device=device)
    elif architecture == "EquiformerV2":
        meta_models = MetaEquiformerV2Models(device=device)
    elif architecture == "ADiT":
        meta_models = MetaADiTModels()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return meta_models


def main():
    """Main function to check memory usage of models."""
    args = get_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Download dataset if not present
    dataset_path = Path(f"{args.datasets_base_path}/{args.split_name}/{args.dataset}")
    if not dataset_path.exists():
        print(f"Dataset {args.dataset} not found at {dataset_path}")
        exit()

    # Create dataloaders (we'll only use the validation loader)
    graph = args.architecture in ["SchNet", "EquiformerV2", "ADiT"]
    _, val_loader = get_dataloaders(
        dataset_paths=[dataset_path],
        train_data_fraction=0.9,
        val_data_fraction=0.1,
        batch_size=args.batch_size,
        architecture=args.architecture,
        seed=1024,
        train_workers=0,
        val_workers=0,
        graph=graph,
    )

    # Initialize meta models collection
    meta_models = init_meta_models(args.architecture, device)

    # Create results container
    memory_results = {
        "architecture": args.architecture,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": []
    }

    # Test each model in the meta_models collection
    for model_idx, model in enumerate(meta_models):
        print(f"\nTesting model {model_idx+1}/{len(meta_models)}")
        model.to(device)
        print(f"Model has {model.num_params} parameters")
        
        model_results = {
            "model_idx": model_idx,
            "num_params": model.num_params,
            "iterations": []
        }

        # Clear memory before starting with this model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Run forward passes and log memory
        model.eval()
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.iterations:
                break
            
            print(f"\nIteration {batch_idx+1}/{args.iterations}")
            
            # Free memory and measure baseline
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log memory before forward pass
            before_stats = log_memory_stats(device)
            print(f"Memory before forward pass: {before_stats['allocated_gb']:.3f} GB allocated")
            
            # Reset peak memory tracking right before the forward pass
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            
            # Perform forward pass
            with torch.no_grad():
                start_time = time.time()
                forward_pass(
                    model=model,
                    batch=batch,
                    graph=graph,
                    training=False,
                    device=device,
                    factorize=False,
                )
                elapsed = time.time() - start_time
            
            # Now get the peak memory that occurred during the forward pass
            # This is the key measurement we want
            peak_stats = log_memory_stats(device)
            print(f"Peak memory during forward pass: {peak_stats['max_allocated_gb']:.3f} GB")
            
            # Clean up memory and measure final state
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log memory after cleanup
            after_stats = log_memory_stats(device)
            print(f"Memory after cleanup: {after_stats['allocated_gb']:.3f} GB allocated")
            print(f"Forward pass time: {elapsed:.4f} seconds")
            
            # Save iteration results
            iteration_result = {
                "iteration": batch_idx,
                "batch_size": batch.num_graphs if hasattr(batch, "num_graphs") else args.batch_size,
                "before_memory_gb": before_stats['allocated_gb'],
                "peak_memory_gb": peak_stats['max_allocated_gb'],
                "after_memory_gb": after_stats['allocated_gb'],
                "forward_pass_time": elapsed,
            }
            model_results["iterations"].append(iteration_result)
        
        # Add model results to the overall results
        memory_results["models"].append(model_results)
        
    # Save results to file
    with open(args.output_file, "w") as f:
        json.dump(memory_results, f, indent=2)

    print(f"\nMemory usage statistics saved to {args.output_file}")


if __name__ == "__main__":
    main()
