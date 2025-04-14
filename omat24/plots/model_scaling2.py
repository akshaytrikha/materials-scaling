import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.random.seed(42)  # For reproducibility

# Set up figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
# Increased width from 14 to 16 for less square-like subfigures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Define model sizes for Transformer and EquiformerV2
transformer_sizes = [10e3, 50e3, 100e3, 500e3, 1e6, 10e6]  # 10K to 10M
equiformer_sizes = [10e3, 50e3, 100e3, 200e3, 500e3, 1e6]  # 10K to 1M

# Generate dataset sizes (number of samples from 10K to 1M)
dataset_sizes = np.logspace(4, 6, 6)  # 6 points from 10^4 to 10^6
dataset_size_labels = ["10K", "50K", "100K", "250K", "500K", "1M"]

# Define color map - using viridis (purple to yellow)
colors = plt.cm.viridis(np.linspace(0, 1, 6))

# Create explicit mock data for Transformer models with revised scaling behavior
# Note: Each row represents a dataset size, each column a model size
transformer_data_by_dataset = [
    # 10K dataset (quickly plateaus with larger models)
    [5.0, 4.7, 4.55, 4.3, 4.25, 4.2],
    # 50K dataset (plateaus a bit later)
    [4.8, 4.4, 4.2, 3.8, 3.7, 3.6],
    # 100K dataset (more improvement with larger models)
    [4.5, 4.0, 3.7, 3.3, 3.1, 3.0],
    # 250K dataset (continues improving longer)
    [4.3, 3.7, 3.3, 2.7, 2.3, 1.95],
    # 500K dataset (steeper improvement with model size)
    [4.1, 3.5, 3.1, 2.3, 1.8, 1.4],
    # 1M dataset (continues improving significantly)
    [3.7, 3.2, 2.7, 1.8, 1.2, 0.8]
]

# Create explicit mock data for EquiformerV2 models (better performance than transformers)
equiformer_data_by_dataset = [
    # 10K dataset (quickly plateaus with larger models)
    [1.2, 1.1, 1.05, 0.97, 0.93, 0.9],
    # 50K dataset (plateaus a bit later)
    [1.15, 1.0, 0.9, 0.83, 0.77, 0.73],
    # 100K dataset (more improvement with larger models)
    [1.1, 0.92, 0.8, 0.7, 0.62, 0.57],
    # 250K dataset (continues improving longer)
    [1.0, 0.82, 0.68, 0.55, 0.45, 0.39],
    # 500K dataset (steeper improvement with model size)
    [0.9, 0.7, 0.55, 0.42, 0.33, 0.28],
    # 1M dataset (continues improving significantly)
    [0.8, 0.62, 0.48, 0.36, 0.29, 0.27]
]

# Plot Transformer Models (by dataset size)
for i, (losses, label) in enumerate(zip(transformer_data_by_dataset, dataset_size_labels)):
    # Plot data points and connect with solid line
    ax1.loglog(transformer_sizes, losses, 'o-', color=colors[i], markersize=8, linewidth=2, label=label)

# Plot EquiformerV2 Models (by dataset size)
for i, (losses, label) in enumerate(zip(equiformer_data_by_dataset, dataset_size_labels)):
    # Plot data points and connect with solid line
    ax2.loglog(equiformer_sizes, losses, 'o-', color=colors[i], markersize=8, linewidth=2, label=label)

# Set up axis labels, titles and legends
ax1.set_xlabel('Model Size (parameters)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Model Size (parameters)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax1.set_title('Transformer Models', fontsize=14, fontweight='bold')
ax2.set_title('EquiformerV2 Models', fontsize=14, fontweight='bold')

# Add legends OUTSIDE the plot area
ax1.legend(title='Dataset Size', loc='upper left', fontsize=9, 
           bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax2.legend(title='Dataset Size', loc='upper left', fontsize=9, 
           bbox_to_anchor=(1.01, 1), borderaxespad=0)

# Set axis limits
ax1.set_xlim(8e3, 1.2e7)
ax1.set_ylim(0.7, 5.3)
ax2.set_xlim(8e3, 1.2e6)
ax2.set_ylim(0.25, 1.3)

# Configure grids
for ax in [ax1, ax2]:
    ax.grid(True, which="major", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    # Format y-axis ticks (appropriate for each subplot)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# Set y-ticks with appropriate values for each subplot
y_ticks_transformer = [0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
y_ticks_equiformer = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
ax1.set_yticks(y_ticks_transformer)
ax2.set_yticks(y_ticks_equiformer)

# Set overall title
fig.suptitle('Loss vs Model and Dataset Size', fontsize=16, fontweight='bold', y=0.98)

# Adjust layout and save - with extra space on right for legends
plt.tight_layout()
plt.subplots_adjust(top=0.88, right=0.85)
plt.savefig('model_scaling2.png', dpi=300, bbox_inches='tight') 