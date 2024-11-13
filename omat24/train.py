# External
import torch
from pathlib import Path
import pprint
import json
from datetime import datetime

# Internal
from data import OMat24Dataset, get_dataloaders
from arg_parser import get_args
from models.fcn import MetaFCNModels
import train_utils.fcn_train_utils as train_utils

# Set seed & device
seed = 1000
torch.manual_seed(seed)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def train_model(model, train_loader, val_loader, args):
    # Initialize optimizer and scheduler
    optimizer = train_utils.get_optimizer(model, learning_rate=args.lr)
    scheduler = train_utils.get_scheduler(optimizer)

    # Train model
    model, losses = train_utils.train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_epochs=args.num_epochs,
        device=DEVICE,
    )

    return model, losses

if __name__ == "__main__":
    args = get_args()

    # Load dataset
    dataset_name = "rattled-300-subsampled"
    dataset_path = Path(f"datasets/{dataset_name}")
    dataset = OMat24Dataset(dataset_path=dataset_path, augment=args.augment)
    train_loader, val_loader = get_dataloaders(
        dataset, data_fraction=args.data_fraction, batch_size=args.batch_size
    )

    # Initialize meta model class
    meta_models = MetaFCNModels(max_atoms=args.max_n_atoms)
    
    # Dictionary to store results for all models
    all_results = {}
    
    # Train each model configuration
    for model_idx, model in enumerate(meta_models):
        print(f"\nTraining Model {model_idx + 1}/{len(meta_models)}")
        print(f"Architecture: embedding_dim={model.embedding_dim}, "
              f"hidden_dim={model.hidden_dim}, depth={model.depth}")
        print(f"Number of parameters: {model.num_params:,}")

        # Train the model
        trained_model, losses = train_model(model, train_loader, val_loader, args)
        
        # Save model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"checkpoints/fcn_model_{model_idx}_{timestamp}.pth"
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': {
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim,
                'depth': model.depth,
                'num_params': model.num_params
            },
            'losses': losses
        }, checkpoint_path)
        
        # Store results
        all_results[f"model_{model_idx}"] = {
            'config': {
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim,
                'depth': model.depth,
                'num_params': model.num_params
            },
            'losses': losses,
            'checkpoint_path': checkpoint_path
        }
    
    # Save all results to JSON
    results_path = f"results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nTraining completed. Results saved to {results_path}")