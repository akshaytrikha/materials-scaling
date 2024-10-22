import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_single_model(metrics, df_key, params_key, output_dir):
    """Create/update plot for a single model's training progress"""
    data = metrics[df_key][params_key]
    epochs = range(len(data['train_loss']))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, data['val_loss'], label='Validation Loss', marker='s')
    plt.axhline(y=data['best_val_loss'], color='r', linestyle='--', 
                label=f'Best Val Loss: {data["best_val_loss"]:.4f}')
    
    plt.title(f'Training Metrics - {df_key}% Data, {params_key} Parameters')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / f'individual_df{df_key}_params{params_key}.png')
    plt.close()

def plot_data_fraction_summary(metrics, df_key, output_dir):
    """Create summary plots for all models in a data fraction"""
    all_train_losses = []
    all_val_losses = []
    model_sizes = []
    
    # Collect data for all models in this data fraction
    for params_key in metrics[df_key].keys():
        data = metrics[df_key][params_key]
        all_train_losses.append(data['train_loss'])
        all_val_losses.append(data['val_loss'])
        model_sizes.append(int(params_key))
        
    epochs = range(len(all_train_losses[0]))
    
    # Combined train loss plot
    plt.figure(figsize=(12, 7))
    for i, params_key in enumerate(metrics[df_key].keys()):
        plt.plot(epochs, all_train_losses[i], 
                label=f'{params_key} params', marker='o')
    
    plt.title(f'Training Loss Comparison - {df_key}% Data')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f'combined_train_df{df_key}.png')
    plt.close()
    
    # Combined validation loss plot
    plt.figure(figsize=(12, 7))
    for i, params_key in enumerate(metrics[df_key].keys()):
        plt.plot(epochs, all_val_losses[i], 
                label=f'{params_key} params', marker='s')
        plt.axhline(y=metrics[df_key][params_key]['best_val_loss'], 
                   color=plt.gca().lines[-1].get_color(), 
                   linestyle='--', 
                   label=f'Best Val ({params_key} params)')
    
    plt.title(f'Validation Loss Comparison - {df_key}% Data')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f'combined_val_df{df_key}.png')
    plt.close()
    
    # Scaling plot
    final_train_losses = [losses[-1] for losses in all_train_losses]
    final_val_losses = [losses[-1] for losses in all_val_losses]
    best_val_losses = [metrics[df_key][str(size)]['best_val_loss'] for size in model_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(model_sizes, final_train_losses, 'o-', label='Final Train Loss')
    plt.semilogx(model_sizes, final_val_losses, 's-', label='Final Val Loss')
    plt.semilogx(model_sizes, best_val_losses, 'D-', label='Best Val Loss')
    
    plt.title(f'Loss vs Model Size - {df_key}% Data')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f'scaling_df{df_key}.png')
    plt.close()

def update_plots(metrics_file, plots_dir="plots", current_df=None):
    """
    Update plots based on current training state.
    If current_df is provided, only update that data fraction's plots.
    """
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if current_df is not None:
        # Update only the current data fraction's plots
        df_key = str(current_df)
        for params_key in metrics[df_key].keys():
            plot_single_model(metrics, df_key, params_key, plots_dir)
        plot_data_fraction_summary(metrics, df_key, plots_dir)
    else:
        # Update all plots
        for df_key in metrics.keys():
            for params_key in metrics[df_key].keys():
                plot_single_model(metrics, df_key, params_key, plots_dir)
            plot_data_fraction_summary(metrics, df_key, plots_dir)

def log_training_metrics(filename, data_fraction, model_params, epoch, train_loss, val_loss, best_val_loss):
    """
    Log training metrics to a JSON file with the following structure:
    {
        "50": {  # data_fraction percentage
            "1000000": {  # model_params
                "train_loss": [0.1, 0.2, ...],  # list of train losses
                "val_loss": [0.15, 0.25, ...],   # list of val losses
                "best_val_loss": float
            }
        }
    }
    """
    filepath = Path(filename)
    if filepath.exists():
        with open(filepath, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    df_key = str(int(data_fraction * 100))
    params_key = str(model_params)
    if df_key not in metrics:
        metrics[df_key] = {}
    if params_key not in metrics[df_key]:
        metrics[df_key][params_key] = {
            "train_loss": [],
            "val_loss": [],
            "best_val_loss": float("inf")
        }
    metrics[df_key][params_key]["train_loss"].append(float(train_loss))
    metrics[df_key][params_key]["val_loss"].append(float(val_loss))
    metrics[df_key][params_key]["best_val_loss"] = best_val_loss
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)