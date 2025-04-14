import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.random.seed(42)  # For reproducibility

# Set up figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
# Increased width from 14 to 16 for less square-like subfigures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Generate dataset sizes (number of samples from 10K to 1M, matching figure1.py)
dataset_sizes = np.logspace(4, 6, 6)  # 6 points from 10^4 to 10^6

# Define model sizes for Transformer and EquiformerV2
transformer_sizes = [10e3, 50e3, 100e3, 500e3, 1e6, 10e6]  # 10K to 10M
transformer_size_labels = ["10K", "50K", "100K", "500K", "1M", "10M"]

equiformer_sizes = [10e3, 50e3, 100e3, 200e3, 500e3, 1e6]  # 10K to 1M
equiformer_size_labels = ["10K", "50K", "100K", "200K", "500K", "1M"]

# Define color map - using viridis (purple to yellow)
colors = plt.cm.viridis(np.linspace(0, 1, 6))

# Create explicit mock data for Transformer models with correct scaling behavior
transformer_data = [
    # 10K parameters model (worst performance, plateaus quickly)
    [5.0, 4.65, 4.45, 4.32, 4.25, 4.2],
    # 50K parameters model (plateaus a bit later)
    [4.8, 4.35, 4.05, 3.85, 3.72, 3.65],
    # 100K parameters model (continues improving longer)
    [4.5, 4.0, 3.65, 3.35, 3.15, 3.0],
    # 500K parameters model (steeper improvement with more data)
    [4.3, 3.7, 3.25, 2.85, 2.5, 2.2],
    # 1M parameters model (continues improving significantly)
    [4.0, 3.3, 2.75, 2.25, 1.8, 1.4],
    # 10M parameters model (keeps improving throughout)
    [3.7, 2.95, 2.3, 1.7, 1.15, 0.8]
]

# Create explicit mock data for EquiformerV2 models (better performance than transformers)
equiformer_data = [
    # 10K parameters model (worst performance, plateaus quickly)
    [1.2, 1.08, 1.0, 0.95, 0.92, 0.9],
    # 50K parameters model (plateaus a bit later)
    [1.15, 1.0, 0.88, 0.81, 0.76, 0.73],
    # 100K parameters model (continues improving longer)
    [1.1, 0.93, 0.8, 0.7, 0.63, 0.58],
    # 200K parameters model (steeper improvement with more data)
    [1.0, 0.83, 0.7, 0.6, 0.52, 0.46],
    # 500K parameters model (continues improving significantly)
    [0.9, 0.72, 0.58, 0.47, 0.39, 0.34],
    # 1M parameters model (keeps improving throughout)
    [0.8, 0.62, 0.48, 0.38, 0.32, 0.27]
]

# Plot Transformer Models
for i, (losses, label) in enumerate(zip(transformer_data, transformer_size_labels)):
    # Plot data points and connect with solid line
    ax1.loglog(dataset_sizes, losses, 'o-', color=colors[i], markersize=8, linewidth=2, label=label)

# Plot EquiformerV2 Models
for i, (losses, label) in enumerate(zip(equiformer_data, equiformer_size_labels)):
    # Plot data points and connect with solid line
    ax2.loglog(dataset_sizes, losses, 'o-', color=colors[i], markersize=8, linewidth=2, label=label)

# Set up axis labels, titles and legends
ax1.set_xlabel('Dataset Size (samples)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dataset Size (samples)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax1.set_title('Transformer Models', fontsize=14, fontweight='bold')
ax2.set_title('EquiformerV2 Models', fontsize=14, fontweight='bold')

# Add legends OUTSIDE the plot area
ax1.legend(title='Params', loc='upper left', fontsize=9, 
           bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax2.legend(title='Params', loc='upper left', fontsize=9, 
           bbox_to_anchor=(1.01, 1), borderaxespad=0)

# Set axis limits
ax1.set_xlim(8e3, 1.2e6)
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
plt.savefig('dataset_scaling2.png', dpi=300, bbox_inches='tight') 