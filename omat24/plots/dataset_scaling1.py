import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
np.random.seed(40)  # For reproducibility

# Set up figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))

# Generate dataset sizes (number of samples from 10K to 1M)
dataset_sizes = np.logspace(4, 6, 7)  # 7 points from 10^4 to 10^6

# Create realistic data with noise
# Base power law for transformer (from ~4 to ~0.9)
transformer_base = 15 * dataset_sizes**(-0.2)
# Add realistic noise (approximately 5-10% variability)
noise_factor_t = 0.08  # 8% noise
transformer_losses = transformer_base * (1 + noise_factor_t * (np.random.random(len(dataset_sizes)) - 0.5))
# Ensure first and last points are close to expected range
transformer_losses[0] = 3.9 + 0.1 * np.random.random()
transformer_losses[-1] = 0.9 - 0.05 * np.random.random()

# Base power law for equiformer (from ~1.5 to ~0.2)
equiformer_base = 6 * dataset_sizes**(-0.22)
# Add realistic noise
noise_factor_e = 0.07  # 7% noise
equiformer_losses = equiformer_base * (1 + noise_factor_e * (np.random.random(len(dataset_sizes)) - 0.5))
# Ensure first and last points are close to expected range
equiformer_losses[0] = 1.45 + 0.1 * np.random.random()
equiformer_losses[-1] = 0.22 - 0.03 * np.random.random()

# Define power law function for fitting
def power_law(x, C, alpha):
    return C * x**(-alpha)

# Fit power laws
transformer_params, _ = curve_fit(power_law, dataset_sizes, transformer_losses)
equiformer_params, _ = curve_fit(power_law, dataset_sizes, equiformer_losses)

# Generate smooth curves for the fitted power laws
x_smooth = np.logspace(4, 6, 100)
transformer_fit = power_law(x_smooth, *transformer_params)
equiformer_fit = power_law(x_smooth, *equiformer_params)

# Create power law equation strings (rounded for display)
transformer_eq = f"{transformer_params[0]:.2f} · N^{-transformer_params[1]:.3f}"
equiformer_eq = f"{equiformer_params[0]:.2f} · N^{-equiformer_params[1]:.3f}"

# Define colors
transformer_color = '#1f77b4'  # Original blue
equiformer_color = '#ff7f0e'  # Original orange

# Create lighter versions for the fit lines
transformer_fit_color = tuple([x + (1-x)*0.5 for x in mcolors.to_rgb(transformer_color)])  # 50% lighter
equiformer_fit_color = tuple([x + (1-x)*0.5 for x in mcolors.to_rgb(equiformer_color)])  # 50% lighter

# Plot data points
plt.loglog(dataset_sizes, transformer_losses, 'o', color=transformer_color, markersize=8, label='Transformer')
plt.loglog(dataset_sizes, equiformer_losses, 's', color=equiformer_color, markersize=8, label='EquiformerV2')

# Add solid lines connecting the actual data points
plt.loglog(dataset_sizes, transformer_losses, '-', color=transformer_color, linewidth=2)
plt.loglog(dataset_sizes, equiformer_losses, '-', color=equiformer_color, linewidth=2)

# Add dashed lines for the power law fits with lighter colors
plt.loglog(x_smooth, transformer_fit, '--', color=transformer_fit_color, linewidth=1.7, 
           label=f'Transformer fit: {transformer_eq}')
plt.loglog(x_smooth, equiformer_fit, '--', color=equiformer_fit_color, linewidth=1.7, 
           label=f'EquiformerV2 fit: {equiformer_eq}')

# Add labels and title
plt.xlabel('Dataset Size (samples)', fontsize=14, fontweight='bold')
plt.ylabel('Test Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)

# Set axis limits
plt.xlim(8e3, 1.2e6)
plt.ylim(0.15, 5)

# Fix y-axis tick labels - with fewer labels
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
# Only show a reasonable number of y-ticks
y_ticks = [0.2, 0.4, 0.7, 1.0, 2.0, 3.0, 4.0, 5.0]
plt.yticks(y_ticks)

# Add grid
plt.grid(True, which="major", ls="-", alpha=0.2)

# Save the plot without displaying it
plt.tight_layout()
plt.savefig('dataset_scaling1.png', dpi=300)

# Print the generated data for reference
print("Dataset sizes:", dataset_sizes)
print("Transformer losses:", transformer_losses)
print("EquiformerV2 losses:", equiformer_losses)
