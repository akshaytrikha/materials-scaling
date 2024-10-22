import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_single_model(metrics, df_key, params_key, output_dir):
    """Create/update plot for a single model's training progress"""
    plt.style.use('default')  # Reset to default style
    
    data = metrics[df_key][params_key]
    epochs = range(len(data['train_loss']))
    
    # Check if model is FCN to include additional parameters
    if "FCN" in data['model_name']:
        title = (f"Training Metrics - {data['model_name']}\n"
                f"Data: {df_key}%, Params: {params_key}\n"
                f"BS={data['batch_size']}, LR={data['learning_rate']}")
    else:
        title = (f"Training Metrics - {data['model_name']}\n"
                f"Data: {df_key}%, Params: {params_key}\n"
                f"BS={data['batch_size']}, LR={data['learning_rate']}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], 'b-o', label='Train Loss', markersize=4)
    plt.plot(epochs, data['val_loss'], 'g-s', label='Validation Loss', markersize=4)
    plt.axhline(y=data['best_val_loss'], color='r', linestyle='--', 
                label=f'Best Val Loss: {data["best_val_loss"]:.4f}')
    
    plt.title(title)
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'individual_df{df_key}_params{params_key}.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_data_fraction_summary(metrics, df_key, output_dir):
    """Create summary plots for all models in a data fraction"""
    plt.style.use('default')  # Reset to default style
    
    all_train_losses = []
    all_val_losses = []
    model_sizes = []
    model_names = []
    max_epochs = 0  # Track the maximum number of epochs
    
    # Collect data and find max epochs
    for params_key in metrics[df_key].keys():
        data = metrics[df_key][params_key]
        train_losses = data['train_loss']
        val_losses = data['val_loss']
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        model_sizes.append(int(params_key))
        model_names.append(data['model_name'])
        max_epochs = max(max_epochs, len(train_losses))
    
    epochs = range(max_epochs)
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_sizes)))
    
    # Combined train loss plot
    plt.figure(figsize=(12, 7))
    for i, (params_key, color) in enumerate(zip(metrics[df_key].keys(), colors)):
        model_epochs = range(len(all_train_losses[i]))
        data = metrics[df_key][params_key]
        plt.plot(model_epochs, all_train_losses[i], 
                color=color, marker='o', markersize=4,
                label=f'{model_names[i]} ({params_key} params)')
    
    title = (f"Training Loss Comparison\n"
            f"Data: {df_key}%, BS={data['batch_size']}, LR={data['learning_rate']}")
    plt.title(title)
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'combined_train_df{df_key}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Combined validation loss plot
    plt.figure(figsize=(12, 7))
    for i, (params_key, color) in enumerate(zip(metrics[df_key].keys(), colors)):
        model_epochs = range(len(all_val_losses[i]))
        data = metrics[df_key][params_key]
        plt.plot(model_epochs, all_val_losses[i], 
                color=color, marker='s', markersize=4,
                label=f'{model_names[i]} ({params_key} params)')
    
    title = (f"Validation Loss Comparison\n"
            f"Data: {df_key}%, BS={data['batch_size']}, LR={data['learning_rate']}")
    plt.title(title)
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'combined_val_df{df_key}.png', dpi=100, bbox_inches='tight')
    plt.close()


def update_plots(metrics_file, plots_dir="plots", current_df=None):
    """
    Update plots based on current training state.
    If current_df is provided, only update that data fraction's plots.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"No metrics file found at {metrics_file}")
        return
    except json.JSONDecodeError:
        print(f"Error reading metrics file at {metrics_file}")
        return
    
    if current_df is not None:
        # Update only the current data fraction's plots
        df_key = str(current_df)
        if df_key in metrics:
            for params_key in metrics[df_key].keys():
                plot_single_model(metrics, df_key, params_key, plots_dir)
            plot_data_fraction_summary(metrics, df_key, plots_dir)
        else:
            print(f"No data found for fraction {current_df}")
    else:
        # Update all plots
        for df_key in metrics.keys():
            for params_key in metrics[df_key].keys():
                plot_single_model(metrics, df_key, params_key, plots_dir)
            plot_data_fraction_summary(metrics, df_key, plots_dir)

def log_training_metrics(filename, data_fraction, model_params, epoch, train_loss, val_loss, best_val_loss, 
                        model_name, batch_size, learning_rate):
    """
    Log training metrics to a JSON file with the following structure:
    {
        "50": {  # data_fraction percentage
            "1000000": {  # model_params
                "train_loss": [0.1, 0.2, ...],  # list of train losses
                "val_loss": [0.15, 0.25, ...],   # list of val losses
                "best_val_loss": float,
                "model_name": str,
                "batch_size": int,
                "learning_rate": float
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
            "best_val_loss": float("inf"),
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    
    metrics[df_key][params_key]["train_loss"].append(float(train_loss))
    metrics[df_key][params_key]["val_loss"].append(float(val_loss))
    metrics[df_key][params_key]["best_val_loss"] = best_val_loss
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)