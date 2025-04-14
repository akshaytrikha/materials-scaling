import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

np.random.seed(40)  # For reproducibility

# Set up figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

# Generate compute values (in PF-days, from 10^-9 to 10^-2)
compute_values = np.logspace(-9, -2, 15)  # 15 points from 10^-9 to 10^-2

# MANUALLY SPECIFIED DATA POINTS - use from compute_scaling1.py
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

# Define colors - USING EXACT SAME COLORS AS compute_scaling1.py
transformer_color = '#1f77b4'  # Original blue
equiformer_color = '#ff7f0e'   # Original orange

# Create lighter versions for the fit lines
transformer_fit_color = tuple([x + (1-x)*0.5 for x in mcolors.to_rgb(transformer_color)])  # 50% lighter
equiformer_fit_color = tuple([x + (1-x)*0.5 for x in mcolors.to_rgb(equiformer_color)])  # 50% lighter

################# TRANSFORMER CURVES (LEFT SUBPLOT) #################
# MANUALLY DEFINE TRAINING CURVES - exactly 10 points each
# Each curve ends at one of the points on the transformer curve

# For reference:
# compute_values[0] = 1.00e-09, loss = 4.00
# compute_values[1] = 3.16e-09, loss = 3.85
# compute_values[2] = 1.00e-08, loss = 3.52
# compute_values[3] = 3.16e-08, loss = 3.32
# compute_values[4] = 1.00e-07, loss = 3.02
# compute_values[5] = 3.16e-07, loss = 2.74
# compute_values[6] = 1.00e-06, loss = 2.46
# compute_values[7] = 3.16e-06, loss = 2.16
# compute_values[8] = 1.00e-05, loss = 1.91
# compute_values[9] = 3.16e-05, loss = 1.65
# compute_values[10] = 1.00e-04, loss = 1.42
# compute_values[11] = 3.16e-04, loss = 1.20
# compute_values[12] = 1.00e-03, loss = 1.05
# compute_values[13] = 3.16e-03, loss = 0.86
# compute_values[14] = 1.00e-02, loss = 0.72

# Curve ending at point 4 (compute = 1.00e-07, loss = 3.02)
curve1_compute = np.array([5.0e-10, 8.0e-10, 1.3e-09, 2.1e-09, 3.5e-09, 6.0e-09, 1.0e-08, 1.8e-08, 5.0e-08, 1.00e-07])
curve1_loss =    np.array([5.0,     4.9,     4.7,     4.6,     4.4,     4.1,     3.8,     3.5,     3.2,     3.02])

# Curve ending at point 6 (compute = 1.00e-06, loss = 2.46)
curve2_compute = np.array([1.0e-09, 2.5e-09, 6.0e-09, 1.4e-08, 2.5e-08, 4.0e-08, 5.5e-08, 2.0e-07, 5.0e-07, 1.00e-06])
curve2_loss =    np.array([5.2,     5.0,     4.6,     4.2,     3.8,     3.5,     3.2,     2.9,     2.6,     2.46])

# Curve ending at point 8 (compute = 1.00e-05, loss = 1.91)
curve3_compute = np.array([1.5e-09, 4.0e-09, 1.0e-08, 3.0e-08, 8.0e-08, 2.0e-07, 4.0e-07, 1.0e-06, 3.0e-06, 1.00e-05])
curve3_loss =    np.array([5.5,     5.2,     4.8,     4.3,     3.8,     3.3,     2.7,     2.3,     2.1,     1.91])

# Curve ending at point 10 (compute = 1.00e-04, loss = 1.42)
curve4_compute = np.array([2.0e-09, 6.0e-09, 2.0e-08, 6.0e-08, 2.0e-07, 6.0e-07, 2.0e-06, 5.0e-06, 3.0e-05, 1.00e-04])
curve4_loss =    np.array([5.6,     5.3,     4.8,     4.3,     3.7,     3.1,     2.5,     2.0,     1.6,     1.42])

# Curve ending at point 12 (compute = 1.00e-03, loss = 1.05)
curve5_compute = np.array([4.0e-09, 1.0e-08, 4.0e-08, 1.0e-07, 4.0e-07, 1.0e-06, 5.0e-06, 1.5e-05, 3.0e-04, 1.00e-03])
curve5_loss =    np.array([5.8,     5.5,     5.0,     4.5,     3.9,     3.3,     2.7,     2.1,     1.5,     1.05])

# Curve ending at point 14 (compute = 1.00e-02, loss = 0.72)
curve6_compute = np.array([5.0e-09, 2.0e-08, 8.0e-08, 3.0e-07, 1.0e-06, 5.0e-06, 2.0e-05, 1.0e-04, 1.0e-03, 1.00e-02])
curve6_loss =    np.array([6.0,     5.6,     5.1,     4.5,     3.8,     3.2,     2.5,     1.9,     1.2,     0.72])

# Curve ending at point 1 (compute = 1.00e-09, loss = 4.00)
curve7_compute = np.array([1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10, 6.0e-10, 7.0e-10, 8.0e-10, 9.0e-10, 1.00e-09])
curve7_loss =    np.array([4.8,     4.7,     4.6,     4.5,     4.4,     4.3,     4.2,     4.1,     4.05,    4.00])

# Curve ending at point 2 (compute = 3.16e-09, loss = 3.85)
curve8_compute = np.array([5.0e-10, 8.0e-10, 1.1e-09, 1.4e-09, 1.7e-09, 2.0e-09, 2.3e-09, 2.6e-09, 2.9e-09, 3.16e-09])
curve8_loss =    np.array([4.6,     4.5,     4.4,     4.3,     4.2,     4.1,     4.0,     3.95,    3.9,     3.85])

# Curve ending at point 3 (compute = 1.00e-08, loss = 3.52)
curve9_compute = np.array([3.0e-10, 6.0e-10, 1.0e-09, 1.5e-09, 2.5e-09, 4.0e-09, 5.5e-09, 7.0e-09, 9.0e-09, 1.00e-08])
curve9_loss =    np.array([4.9,     4.8,     4.7,     4.5,     4.3,     4.1,     3.9,     3.7,     3.6,     3.52])

# Curve ending at point 5 (compute = 3.16e-07, loss = 2.74)
curve10_compute = np.array([8.0e-10, 2.0e-09, 5.0e-09, 1.0e-08, 3.0e-08, 6.0e-08, 1.0e-07, 1.8e-07, 2.5e-07, 3.16e-07])
curve10_loss =    np.array([5.3,     5.1,     4.8,     4.4,     4.0,     3.6,     3.3,     3.0,     2.8,     2.74])

# Curve ending at point 7 (compute = 3.16e-06, loss = 2.16)
curve11_compute = np.array([1.2e-09, 3.0e-09, 8.0e-09, 2.0e-08, 5.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 2.0e-06, 3.16e-06])
curve11_loss =    np.array([5.4,     5.2,     4.9,     4.5,     4.0,     3.5,     3.0,     2.6,     2.3,     2.16])

# Curve ending at point 9 (compute = 3.16e-05, loss = 1.65)
curve12_compute = np.array([1.8e-09, 5.0e-09, 1.5e-08, 4.0e-08, 1.0e-07, 3.0e-07, 8.0e-07, 3.0e-06, 1.0e-05, 3.16e-05])
curve12_loss =    np.array([5.5,     5.3,     4.9,     4.4,     3.9,     3.4,     2.8,     2.3,     1.9,     1.65])

# Curve ending at point 11 (compute = 3.16e-04, loss = 1.20)
curve13_compute = np.array([2.5e-09, 8.0e-09, 2.5e-08, 8.0e-08, 2.5e-07, 8.0e-07, 3.0e-06, 1.0e-05, 1.0e-04, 3.16e-04])
curve13_loss =    np.array([5.7,     5.4,     4.9,     4.4,     3.8,     3.2,     2.6,     2.0,     1.5,     1.20])

# Curve ending at point 13 (compute = 3.16e-03, loss = 0.86)
curve14_compute = np.array([7.0e-09, 3.0e-08, 9.0e-08, 3.0e-07, 9.0e-07, 3.0e-06, 9.0e-06, 3.0e-05, 5.0e-04, 3.16e-03])
curve14_loss =    np.array([6.2,     5.8,     5.2,     4.6,     3.9,     3.2,     2.5,     1.8,     1.2,     0.86])

# Higher starting point curves ending at different points
curve15_compute = np.array([2.0e-09, 6.0e-09, 1.5e-08, 4.0e-08, 1.0e-07, 2.5e-07, 6.0e-07, 1.5e-06, 5.0e-06, 1.00e-05])
curve15_loss =    np.array([6.5,     6.0,     5.5,     5.0,     4.5,     3.8,     3.2,     2.6,     2.2,     1.91])

curve16_compute = np.array([5.0e-09, 1.5e-08, 5.0e-08, 1.5e-07, 5.0e-07, 1.5e-06, 5.0e-06, 1.5e-05, 5.0e-05, 1.00e-04])
curve16_loss =    np.array([6.8,     6.3,     5.7,     5.0,     4.3,     3.6,     2.9,     2.3,     1.8,     1.42])

curve17_compute = np.array([3.0e-09, 1.0e-08, 3.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.00e-04])
curve17_loss =    np.array([6.3,     5.9,     5.4,     4.8,     4.2,     3.5,     2.9,     2.3,     1.8,     1.42])

curve18_compute = np.array([1.0e-08, 3.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.0e-04, 3.16e-04])
curve18_loss =    np.array([6.0,     5.5,     5.0,     4.5,     3.8,     3.2,     2.6,     2.0,     1.5,     1.20])

curve19_compute = np.array([3.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.0e-04, 3.0e-04, 1.00e-03])
curve19_loss =    np.array([5.8,     5.2,     4.7,     4.0,     3.4,     2.8,     2.2,     1.8,     1.4,     1.05])

curve20_compute = np.array([1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.0e-04, 3.0e-04, 1.0e-03, 3.16e-03])
curve20_loss =    np.array([5.5,     5.0,     4.5,     3.8,     3.2,     2.6,     2.0,     1.6,     1.2,     0.86])

# Store all curves in lists for easy plotting
all_curves_compute = [curve1_compute, curve2_compute, curve3_compute, curve4_compute, curve5_compute,
                     curve6_compute, curve7_compute, curve8_compute, curve9_compute, curve10_compute,
                     curve11_compute, curve12_compute, curve13_compute, curve14_compute, curve15_compute,
                     curve16_compute, curve17_compute, curve18_compute, curve19_compute, curve20_compute]

all_curves_loss = [curve1_loss, curve2_loss, curve3_loss, curve4_loss, curve5_loss,
                  curve6_loss, curve7_loss, curve8_loss, curve9_loss, curve10_loss,
                  curve11_loss, curve12_loss, curve13_loss, curve14_loss, curve15_loss,
                  curve16_loss, curve17_loss, curve18_loss, curve19_loss, curve20_loss]

################# EQUIFORMER CURVES (RIGHT SUBPLOT) #################
# MANUALLY DEFINE TRAINING CURVES - exactly 10 points each
# Each curve ends at one of the points on the equiformer curve

# For reference:
# compute_values[0] = 1.00e-09, loss = 1.40
# compute_values[1] = 3.16e-09, loss = 1.32
# compute_values[2] = 1.00e-08, loss = 1.14
# compute_values[3] = 3.16e-08, loss = 1.06
# compute_values[4] = 1.00e-07, loss = 0.95
# compute_values[5] = 3.16e-07, loss = 0.82
# compute_values[6] = 1.00e-06, loss = 0.71
# compute_values[7] = 3.16e-06, loss = 0.67
# compute_values[8] = 1.00e-05, loss = 0.53
# compute_values[9] = 3.16e-05, loss = 0.49
# compute_values[10] = 1.00e-04, loss = 0.41
# compute_values[11] = 3.16e-04, loss = 0.37
# compute_values[12] = 1.00e-03, loss = 0.32
# compute_values[13] = 3.16e-03, loss = 0.26
# compute_values[14] = 1.00e-02, loss = 0.22

# Curve ending at point 4 (compute = 1.00e-07, loss = 0.95)
eq_curve1_compute = np.array([5.0e-10, 8.0e-10, 1.3e-09, 2.1e-09, 3.5e-09, 6.0e-09, 1.0e-08, 1.8e-08, 5.0e-08, 1.00e-07])
eq_curve1_loss =    np.array([1.8,     1.75,    1.7,     1.65,    1.55,    1.45,    1.35,    1.2,     1.05,    0.95])

# Curve ending at point 6 (compute = 1.00e-06, loss = 0.71)
eq_curve2_compute = np.array([1.0e-09, 2.5e-09, 6.0e-09, 1.4e-08, 2.5e-08, 4.0e-08, 5.5e-08, 2.0e-07, 5.0e-07, 1.00e-06])
eq_curve2_loss =    np.array([1.85,    1.75,    1.65,    1.5,     1.35,    1.2,     1.1,     0.95,    0.8,     0.71])

# Curve ending at point 8 (compute = 1.00e-05, loss = 0.53)
eq_curve3_compute = np.array([1.5e-09, 4.0e-09, 1.0e-08, 3.0e-08, 8.0e-08, 2.0e-07, 4.0e-07, 1.0e-06, 3.0e-06, 1.00e-05])
eq_curve3_loss =    np.array([1.9,     1.8,     1.7,     1.5,     1.3,     1.1,     0.9,     0.75,    0.62,    0.53])

# Curve ending at point 10 (compute = 1.00e-04, loss = 0.41)
eq_curve4_compute = np.array([2.0e-09, 6.0e-09, 2.0e-08, 6.0e-08, 2.0e-07, 6.0e-07, 2.0e-06, 5.0e-06, 3.0e-05, 1.00e-04])
eq_curve4_loss =    np.array([1.95,    1.85,    1.7,     1.5,     1.3,     1.05,    0.85,    0.7,     0.52,    0.41])

# Curve ending at point 12 (compute = 1.00e-03, loss = 0.32)
eq_curve5_compute = np.array([4.0e-09, 1.0e-08, 4.0e-08, 1.0e-07, 4.0e-07, 1.0e-06, 5.0e-06, 1.5e-05, 3.0e-04, 1.00e-03])
eq_curve5_loss =    np.array([2.0,     1.9,     1.7,     1.5,     1.3,     1.1,     0.9,     0.7,     0.45,    0.32])

# Curve ending at point 14 (compute = 1.00e-02, loss = 0.22)
eq_curve6_compute = np.array([5.0e-09, 2.0e-08, 8.0e-08, 3.0e-07, 1.0e-06, 5.0e-06, 2.0e-05, 1.0e-04, 1.0e-03, 1.00e-02])
eq_curve6_loss =    np.array([2.0,     1.85,    1.7,     1.5,     1.3,     1.05,    0.85,    0.65,    0.4,     0.22])

# Curve ending at point 1 (compute = 1.00e-09, loss = 1.40)
eq_curve7_compute = np.array([1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10, 6.0e-10, 7.0e-10, 8.0e-10, 9.0e-10, 1.00e-09])
eq_curve7_loss =    np.array([1.9,     1.85,    1.8,     1.7,     1.65,    1.6,     1.55,    1.5,     1.45,    1.40])

# Curve ending at point 2 (compute = 3.16e-09, loss = 1.32)
eq_curve8_compute = np.array([5.0e-10, 8.0e-10, 1.1e-09, 1.4e-09, 1.7e-09, 2.0e-09, 2.3e-09, 2.6e-09, 2.9e-09, 3.16e-09])
eq_curve8_loss =    np.array([1.7,     1.65,    1.6,     1.55,    1.5,     1.45,    1.4,     1.37,    1.35,    1.32])

# Curve ending at point 3 (compute = 1.00e-08, loss = 1.14)
eq_curve9_compute = np.array([3.0e-10, 6.0e-10, 1.0e-09, 1.5e-09, 2.5e-09, 4.0e-09, 5.5e-09, 7.0e-09, 9.0e-09, 1.00e-08])
eq_curve9_loss =    np.array([1.75,    1.7,     1.65,    1.55,    1.45,    1.35,    1.25,    1.2,     1.17,    1.14])

# Curve ending at point 5 (compute = 3.16e-07, loss = 0.82)
eq_curve10_compute = np.array([8.0e-10, 2.0e-09, 5.0e-09, 1.0e-08, 3.0e-08, 6.0e-08, 1.0e-07, 1.8e-07, 2.5e-07, 3.16e-07])
eq_curve10_loss =    np.array([1.8,     1.7,     1.6,     1.5,     1.4,     1.25,    1.1,     0.95,    0.88,    0.82])

# Curve ending at point 7 (compute = 3.16e-06, loss = 0.67)
eq_curve11_compute = np.array([1.2e-09, 3.0e-09, 8.0e-09, 2.0e-08, 5.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 2.0e-06, 3.16e-06])
eq_curve11_loss =    np.array([1.85,    1.8,     1.65,    1.5,     1.35,    1.2,     1.0,     0.85,    0.73,    0.67])

# Curve ending at point 9 (compute = 3.16e-05, loss = 0.49)
eq_curve12_compute = np.array([1.8e-09, 5.0e-09, 1.5e-08, 4.0e-08, 1.0e-07, 3.0e-07, 8.0e-07, 3.0e-06, 1.0e-05, 3.16e-05])
eq_curve12_loss =    np.array([1.9,     1.8,     1.65,    1.5,     1.3,     1.15,    0.95,    0.75,    0.6,     0.49])

# Curve ending at point 11 (compute = 3.16e-04, loss = 0.37)
eq_curve13_compute = np.array([2.5e-09, 8.0e-09, 2.5e-08, 8.0e-08, 2.5e-07, 8.0e-07, 3.0e-06, 1.0e-05, 1.0e-04, 3.16e-04])
eq_curve13_loss =    np.array([1.9,     1.8,     1.65,    1.5,     1.3,     1.1,     0.9,     0.7,     0.5,     0.37])

# Curve ending at point 13 (compute = 3.16e-03, loss = 0.26)
eq_curve14_compute = np.array([7.0e-09, 3.0e-08, 9.0e-08, 3.0e-07, 9.0e-07, 3.0e-06, 9.0e-06, 3.0e-05, 5.0e-04, 3.16e-03])
eq_curve14_loss =    np.array([1.95,    1.8,     1.6,     1.4,     1.2,     1.0,     0.8,     0.6,     0.4,     0.26])

# Higher starting point curves ending at different points
eq_curve15_compute = np.array([2.0e-09, 6.0e-09, 1.5e-08, 4.0e-08, 1.0e-07, 2.5e-07, 6.0e-07, 1.5e-06, 5.0e-06, 1.00e-05])
eq_curve15_loss =    np.array([2.1,     1.95,    1.8,     1.65,    1.45,    1.25,    1.05,    0.85,    0.65,    0.53])

eq_curve16_compute = np.array([5.0e-09, 1.5e-08, 5.0e-08, 1.5e-07, 5.0e-07, 1.5e-06, 5.0e-06, 1.5e-05, 5.0e-05, 1.00e-04])
eq_curve16_loss =    np.array([2.2,     2.0,     1.8,     1.6,     1.4,     1.2,     0.95,    0.75,    0.55,    0.41])

eq_curve17_compute = np.array([3.0e-09, 1.0e-08, 3.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.00e-04])
eq_curve17_loss =    np.array([2.15,    1.95,    1.75,    1.55,    1.35,    1.15,    0.95,    0.75,    0.55,    0.41])

eq_curve18_compute = np.array([1.0e-08, 3.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.0e-04, 3.16e-04])
eq_curve18_loss =    np.array([2.05,    1.9,     1.7,     1.5,     1.3,     1.1,     0.85,    0.65,    0.47,    0.37])

eq_curve19_compute = np.array([3.0e-08, 1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.0e-04, 3.0e-04, 1.00e-03])
eq_curve19_loss =    np.array([1.9,     1.7,     1.5,     1.35,    1.15,    0.9,     0.7,     0.55,    0.42,    0.32])

eq_curve20_compute = np.array([1.0e-07, 3.0e-07, 1.0e-06, 3.0e-06, 1.0e-05, 3.0e-05, 1.0e-04, 3.0e-04, 1.0e-03, 3.16e-03])
eq_curve20_loss =    np.array([1.85,    1.65,    1.45,    1.25,    1.05,    0.85,    0.65,    0.5,     0.35,    0.26])

# Store all curves in lists for easy plotting
eq_all_curves_compute = [
    eq_curve1_compute, eq_curve2_compute, eq_curve3_compute, eq_curve4_compute, eq_curve5_compute,
    eq_curve6_compute, eq_curve7_compute, eq_curve8_compute, eq_curve9_compute, eq_curve10_compute,
    eq_curve11_compute, eq_curve12_compute, eq_curve13_compute, eq_curve14_compute, eq_curve15_compute,
    eq_curve16_compute, eq_curve17_compute, eq_curve18_compute, eq_curve19_compute, eq_curve20_compute
]

eq_all_curves_loss = [
    eq_curve1_loss, eq_curve2_loss, eq_curve3_loss, eq_curve4_loss, eq_curve5_loss,
    eq_curve6_loss, eq_curve7_loss, eq_curve8_loss, eq_curve9_loss, eq_curve10_loss,
    eq_curve11_loss, eq_curve12_loss, eq_curve13_loss, eq_curve14_loss, eq_curve15_loss,
    eq_curve16_loss, eq_curve17_loss, eq_curve18_loss, eq_curve19_loss, eq_curve20_loss
]

################# PLOTTING #################
# FIRST SUBPLOT - TRANSFORMER
# Plot all training curves as light blue lines
for compute, loss in zip(all_curves_compute, all_curves_loss):
    ax1.loglog(compute, loss, '-', color='lightblue', linewidth=1, alpha=0.8)

# Plot the transformer data points and line
ax1.loglog(compute_values, transformer_losses, 'o', color=transformer_color, markersize=8, label='Transformer')
ax1.loglog(compute_values, transformer_losses, '-', color=transformer_color, linewidth=2)

# Add dashed line for the power law fit
ax1.loglog(x_smooth, transformer_fit, '--', color=transformer_fit_color, linewidth=1.7,
         label=f'Transformer fit: {transformer_eq}')

# Add legend, labels, and settings for first subplot
ax1.legend(fontsize=10)
ax1.set_xlabel('Compute (PF-days)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Test Loss', fontsize=14, fontweight='bold')
ax1.set_xlim(5e-10, 2e-2)
ax1.set_ylim(0.4, 5.5)
ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
y_ticks1 = [0.4, 0.7, 1.0, 2.0, 3.0, 4.0, 5.0]
ax1.set_yticks(y_ticks1)
ax1.grid(True, which="major", ls="-", alpha=0.2)

# SECOND SUBPLOT - EQUIFORMER
# Plot all training curves as light orange lines
for compute, loss in zip(eq_all_curves_compute, eq_all_curves_loss):
    ax2.loglog(compute, loss, '-', color='bisque', linewidth=1, alpha=0.8)

# Plot the equiformer data points and line
ax2.loglog(compute_values, equiformer_losses, 's', color=equiformer_color, markersize=8, label='EquiformerV2')
ax2.loglog(compute_values, equiformer_losses, '-', color=equiformer_color, linewidth=2)

# Add dashed line for the power law fit
ax2.loglog(x_smooth, equiformer_fit, '--', color=equiformer_fit_color, linewidth=1.7,
         label=f'EquiformerV2 fit: {equiformer_eq}')

# Add legend, labels, and settings for second subplot
ax2.legend(fontsize=10)
ax2.set_xlabel('Compute (PF-days)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Test Loss', fontsize=14, fontweight='bold')
ax2.set_xlim(5e-10, 2e-2)
ax2.set_ylim(0.15, 2.3)  # Different y scale for EquiformerV2
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
y_ticks2 = [0.2, 0.3, 0.4, 0.7, 1.0, 1.5, 2.0]
ax2.set_yticks(y_ticks2)
ax2.grid(True, which="major", ls="-", alpha=0.2)

# Save the plot without displaying it
plt.tight_layout()
plt.savefig('compute_scaling2.png', dpi=300)
