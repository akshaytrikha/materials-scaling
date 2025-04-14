import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
np.random.seed(40)  # For reproducibility

# Set up figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))

# Generate compute values (in PF-days, from 10^-9 to 10^-2)
compute_values = np.logspace(-9, -2, 15)  # 15 points from 10^-9 to 10^-2

# MANUALLY SPECIFIED DATA POINTS - no functions, just explicit values
# Transformer: smooth progression from 4.0 to 0.72 (EXACTLY 15 POINTS)
transformer_losses = np.array([
    4.0,       # 10^-9 (fixed first point)
    3.85,      
    3.52,      
    3.32,      
    3.02,      
    2.74,      
    2.46,      
    2.16,      
    1.91,      
    1.65,      
    1.42,      
    1.20,      
    1.05,      
    0.86,      
    0.72       # 10^-2 (fixed last point)
])

# EquiformerV2: smooth progression from 1.4 to 0.22 (EXACTLY 15 POINTS)
equiformer_losses = np.array([
    1.4,       # 10^-9 (fixed first point)
    1.32,      
    1.14,      
    1.06,      
    0.95,      
    0.82,      
    0.71,      
    0.67,      
    0.53,      
    0.49,      
    0.41,      
    0.37,      
    0.32,      
    0.26,      
    0.22       # 10^-2 (fixed last point)
])

# Define power law function for fitting only (not for data generation)
def power_law(x, C, alpha):
    return C * x**(-alpha)

# Fit power laws to the manually specified points
transformer_params, _ = curve_fit(power_law, compute_values, transformer_losses)
equiformer_params, _ = curve_fit(power_law, compute_values, equiformer_losses)

# Generate smooth curves for the fitted power laws
x_smooth = np.logspace(-9, -2, 100)
transformer_fit = power_law(x_smooth, *transformer_params)
equiformer_fit = power_law(x_smooth, *equiformer_params)

# Create power law equation strings (rounded for display)
transformer_eq = f"{transformer_params[0]:.2f} · C^{-transformer_params[1]:.3f}"
equiformer_eq = f"{equiformer_params[0]:.2f} · C^{-equiformer_params[1]:.3f}"

# Define colors
transformer_color = '#1f77b4'  # Original blue
equiformer_color = '#ff7f0e'  # Original orange

# Create lighter versions for the fit lines
transformer_fit_color = tuple([x + (1-x)*0.5 for x in mcolors.to_rgb(transformer_color)])  # 50% lighter
equiformer_fit_color = tuple([x + (1-x)*0.5 for x in mcolors.to_rgb(equiformer_color)])  # 50% lighter

# Plot data points
plt.loglog(compute_values, transformer_losses, 'o', color=transformer_color, markersize=8, label='Transformer')
plt.loglog(compute_values, equiformer_losses, 's', color=equiformer_color, markersize=8, label='EquiformerV2')

# Add solid lines connecting the actual data points
plt.loglog(compute_values, transformer_losses, '-', color=transformer_color, linewidth=2)
plt.loglog(compute_values, equiformer_losses, '-', color=equiformer_color, linewidth=2)

# Add dashed lines for the power law fits with lighter colors
plt.loglog(x_smooth, transformer_fit, '--', color=transformer_fit_color, linewidth=1.7, 
           label=f'Transformer fit: {transformer_eq}')
plt.loglog(x_smooth, equiformer_fit, '--', color=equiformer_fit_color, linewidth=1.7, 
           label=f'EquiformerV2 fit: {equiformer_eq}')

# Add labels and title
plt.xlabel('Compute (PF-days)', fontsize=14, fontweight='bold')
plt.ylabel('Test Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)

# Set axis limits
plt.xlim(5e-10, 2e-2)
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
plt.savefig('compute_scaling1.png', dpi=300)

# Print the generated data for reference
print("Compute values (PF-days):", compute_values)
print("Transformer losses:", transformer_losses)
print("EquiformerV2 losses:", equiformer_losses) 