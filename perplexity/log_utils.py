import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def setup_plot_style():
    """Set consistent style for all plots"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150
    })

def plot_single_model(metrics, df_key, params_key, output_dir):
    """Create/update plot for a single model's training progress"""
    setup_plot_style()
    data = metrics[df_key][params_key]
    epochs = range(len(data['train_loss']))
    
    if "FCN" in data['model_name']:
        title = (f"Training Metrics - {data['model_name']}\n"
                f"Data: {df_key}%, Params: {params_key}\n"
                f"BS={data['batch_size']}, LR={data['learning_rate']}")
    else:
        title = (f"Training Metrics - {data['model_name']}\n"
                f"Data: {df_key}%, Params: {params_key}\n"
                f"BS={data['batch_size']}, LR={data['learning_rate']}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, data['train_loss'], color='#1f77b4', marker='o', markersize=3, 
            label='Train Loss', markerfacecolor='white', markeredgewidth=1)
    plt.plot(epochs, data['val_loss'], color='#ff7f0e', marker='s', markersize=3,
            label='Validation Loss', markerfacecolor='white', markeredgewidth=1)
    plt.axhline(y=data['best_val_loss'], color='#C82423', linestyle='--', linewidth=1,
            label=f'Best Val Loss: {data["best_val_loss"]:.4f}')
    plt.axhline(y=10.4756, color='b', linestyle='--', label='Best n-Gram Loss')
    
    plt.title(title, pad=10)
    plt.xlabel('Epoch', labelpad=8)
    plt.ylabel('Loss', labelpad=8)
    plt.legend(frameon=False)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'individual_df{df_key}_params{params_key}.png', bbox_inches='tight')
    plt.close()

def plot_data_fraction_summary(metrics, df_key, output_dir):
    """Create summary plots for all models in a data fraction"""
    setup_plot_style()
    
    all_train_losses = []
    all_val_losses = []
    model_sizes = []
    model_names = []
    max_epochs = 0
    
    # Define a set of distinct markers
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8']
    
    for params_key in metrics[df_key].keys():
        data = metrics[df_key][params_key]
        train_losses = data['train_loss']
        val_losses = data['val_loss']
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        model_sizes.append(int(params_key))
        model_names.append(data['model_name'])
        max_epochs = max(max_epochs, len(train_losses))

    # Create color gradients - handle single model case
    num_models = len(model_sizes)
    if num_models == 1:
        train_colors = [plt.cm.Blues(0.6)]
        val_colors = [plt.cm.Oranges(0.6)]
    else:
        train_colors = [plt.cm.Blues(0.5 + 0.5 * i/(num_models-1)) for i in range(num_models)]
        val_colors = [plt.cm.Oranges(0.5 + 0.5 * i/(num_models-1)) for i in range(num_models)]
    
    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[1, 1])
    
    legend_handles = []  # Store handles for combined legend
    
    # Training loss subplot
    for i, (params_key, color, marker) in enumerate(zip(metrics[df_key].keys(), train_colors, markers[:num_models])):
        model_epochs = range(len(all_train_losses[i]))
        line = ax1.plot(model_epochs, all_train_losses[i], 
                       color=color, marker=marker, markersize=4,
                       markerfacecolor='white', markeredgewidth=1,
                       label=f'{params_key} params')[0]
        
        if i == 0:  # Add a label for "Training" to the legend
            legend_handles.append(plt.Line2D([], [], color=train_colors[0], label='Training',
                                          linestyle='-', marker='None'))
        
        # Add model size to legend with just the marker
        legend_handles.append(plt.Line2D([], [], color='gray', marker=marker,
                                       label=f'{params_key} params', linestyle='None',
                                       markerfacecolor='white', markersize=4))
    
    ax1.set_title(f"Loss Comparison - Data: {df_key}%, BS={data['batch_size']}, LR={data['learning_rate']}", 
                 pad=10)
    ax1.set_ylabel('Training Loss', labelpad=8)
    ax1.grid(True, alpha=0.3)
    
    # Validation loss subplot
    for i, (params_key, color, marker) in enumerate(zip(metrics[df_key].keys(), val_colors, markers[:num_models])):
        model_epochs = range(len(all_val_losses[i]))
        line = ax2.plot(model_epochs, all_val_losses[i], 
                       color=color, marker=marker, markersize=4,
                       markerfacecolor='white', markeredgewidth=1)[0]
        
        if i == 0:  # Add a label for "Validation" to the legend
            legend_handles.append(plt.Line2D([], [], color=val_colors[0], label='Validation',
                                          linestyle='-', marker='None'))
    
    ax2.set_xlabel('Epoch', labelpad=8)
    ax2.set_ylabel('Validation Loss', labelpad=8)
    ax2.grid(True, alpha=0.3)
    
    # Add combined legend below the plots
    fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=num_models + 1, frameon=False, fontsize=8)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig(output_dir / f'combined_loss_df{df_key}.png', bbox_inches='tight')
    plt.close()

# update_plots and log_training_metrics functions remain exactly the same
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
        df_key = str(current_df)
        if df_key in metrics:
            for params_key in metrics[df_key].keys():
                plot_single_model(metrics, df_key, params_key, plots_dir)
            plot_data_fraction_summary(metrics, df_key, plots_dir)
        else:
            print(f"No data found for fraction {current_df}")
    else:
        for df_key in metrics.keys():
            for params_key in metrics[df_key].keys():
                plot_single_model(metrics, df_key, params_key, plots_dir)
            plot_data_fraction_summary(metrics, df_key, plots_dir)

def log_training_metrics(filename, data_fraction, model_params, epoch, train_loss, val_loss, best_val_loss, 
                        model_name, batch_size, learning_rate):
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