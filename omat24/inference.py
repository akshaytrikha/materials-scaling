import torch
from pathlib import Path
from tqdm import tqdm
import json
import os

# Internal
from data import get_dataloaders
from data_utils import VALID_DATASETS
from arg_parser import get_args
from models.fcn import MetaFCNModels
from models.transformer_models import MetaTransformerModels
from models.schnet import MetaSchNetModels
from loss import compute_loss

def load_model(checkpoint_path, n_elements):
    """Load the model based on the checkpoint filename."""
    meta_model_name = Path(checkpoint_path).stem.split('_')[0]  # Extract architecture name
    
    if meta_model_name == "FCN":
        meta_model = MetaFCNModels()
    elif meta_model_name == "Transformer":
        meta_model = MetaTransformerModels(
            vocab_size=n_elements,
            max_seq_len=168,
            use_factorized=False,
        )
    elif meta_model_name == "SchNet":
        meta_model = MetaSchNetModels()
    else:
        raise ValueError(f"Unknown model type: {meta_model_name}")
    model = meta_model[0]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_combined_dataloader():
    """Load all validation datasets into one dataloader."""
    dataloaders = [get_dataloaders(dataset, split='val') for dataset in VALID_DATASETS]
    combined_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([dl.dataset for dl in dataloaders]),
        batch_size=32, shuffle=False
    )
    return combined_dataloader

def run_inference(checkpoint_path, n_elements):
    """Run inference on a given checkpoint and store test loss."""
    model = load_model(checkpoint_path, n_elements)
    dataloader = get_combined_dataloader()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            inputs, targets = batch
            predictions = model(inputs)
            loss = compute_loss(predictions, targets)
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
    
    avg_loss = total_loss / total_samples
    
    # Save results
    results_path = "inference_results.json"
    results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    results[checkpoint_path] = avg_loss
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Inference complete for {checkpoint_path}. Test loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    args = get_args()
    checkpoint_path = args.checkpoint  # Assume checkpoint path is passed as an argument
    n_elements = args.n_elements
    run_inference(checkpoint_path, n_elements)