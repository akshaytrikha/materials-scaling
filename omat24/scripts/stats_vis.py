import json
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create stats directory if it doesn't exist
os.makedirs("stats", exist_ok=True)

# Load the data
with open("dataset_stats.json", "r") as f:
    data = json.load(f)

# Get all dataset names
train_datasets = list(data["train"].keys())
val_datasets = list(data["val"].keys())

# Create more readable dataset names for plotting
def clean_dataset_name(name):
    # Replace hyphens with spaces and capitalize words
    parts = name.split('-')
    if len(parts) >= 2 and parts[0] == "rattled":
        if parts[1].isdigit():
            return f"Rattled {parts[1]}"
        elif parts[1] == "relax":
            return "Rattled Relax"
        elif parts[1] == "1000" and len(parts) > 2 and parts[2] == "subsampled":
            return "Rattled 1000 Sub"
        elif parts[1] == "500" and len(parts) > 2 and parts[2] == "subsampled":
            return "Rattled 500 Sub"
        elif parts[1] == "300" and len(parts) > 2 and parts[2] == "subsampled":
            return "Rattled 300 Sub"
    elif parts[0] == "aimd":
        if "3000-npt" in name:
            return "AIMD 3000 NPT"
        elif "3000-nvt" in name:
            return "AIMD 3000 NVT"
        elif "1000-npt" in name:
            return "AIMD 1000 NPT"
        elif "1000-nvt" in name:
            return "AIMD 1000 NVT"
    
    # If no specific rule matches, just return with spaces
    return name.replace('-', ' ').title()

# 1. Energy Visualization for Train and Val Splits
def plot_energy(split):
    datasets = train_datasets if split == "train" else val_datasets
    
    # Extract energy data
    means = [data[split][dataset]["means"]["energy"] for dataset in datasets]
    stds = [data[split][dataset]["std"]["energy"] for dataset in datasets]
    
    # Clean dataset names for plotting
    clean_names = [clean_dataset_name(name) for name in datasets]
    
    # Sort by mean energy values for better visualization
    sorted_indices = np.argsort(means)
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    clean_names = [clean_names[i] for i in sorted_indices]
    
    # Create the plot with Plotly
    fig = go.Figure()
    
    # Add error bars (centered at the mean)
    fig.add_trace(go.Scatter(
        x=clean_names,
        y=means,
        error_y=dict(
            type='data', 
            array=stds, 
            color='rgba(55,55,200,0.8)', 
            thickness=1.5,
            width=8
        ),
        mode='markers',
        marker=dict(
            color='rgba(135,206,250,0.7)',
            size=10,
            line=dict(
                color='rgba(0,0,0,1)',
                width=1
            )
        ),
        name='Energy Mean ± Std',
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f}<br>Std Dev: %{error_y.array:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Energy Mean and Std Dev - {split.capitalize()} Split",
        xaxis_title="Dataset",
        yaxis_title="Energy (mean)",
        xaxis_tickangle=-45,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
        width=1000
    )
    
    # Save the figure (PNG only)
    fig.write_image(f"stats/energy_{split}.png", scale=2)
    
    print(f"Energy plot for {split} split created in stats folder")

# 2. Forces Visualization - simplified to match energy plot style
def plot_forces(split):
    datasets = train_datasets if split == "train" else val_datasets
    
    # Extract force data (3 components: x, y, z)
    means_x = [data[split][dataset]["means"]["forces"][0] for dataset in datasets]
    means_y = [data[split][dataset]["means"]["forces"][1] for dataset in datasets]
    means_z = [data[split][dataset]["means"]["forces"][2] for dataset in datasets]
    
    stds_x = [data[split][dataset]["std"]["forces"][0] for dataset in datasets]
    stds_y = [data[split][dataset]["std"]["forces"][1] for dataset in datasets]
    stds_z = [data[split][dataset]["std"]["forces"][2] for dataset in datasets]
    
    # Clean dataset names for plotting
    clean_names = [clean_dataset_name(name) for name in datasets]
    
    # Sort datasets by name for consistency
    sorted_indices = np.argsort(clean_names)
    clean_names = [clean_names[i] for i in sorted_indices]
    means_x = [means_x[i] for i in sorted_indices]
    means_y = [means_y[i] for i in sorted_indices]
    means_z = [means_z[i] for i in sorted_indices]
    stds_x = [stds_x[i] for i in sorted_indices]
    stds_y = [stds_y[i] for i in sorted_indices]
    stds_z = [stds_z[i] for i in sorted_indices]
    
    # Create figure for forces
    fig = go.Figure()
    
    # Colors for the three components
    colors = ['rgba(52, 152, 219, 0.7)', 'rgba(46, 204, 113, 0.7)', 'rgba(231, 76, 60, 0.7)']
    
    # Add traces for each force component (with error bars)
    fig.add_trace(
        go.Scatter(
            x=clean_names, 
            y=means_x, 
            error_y=dict(type='data', array=stds_x, color='rgba(52, 152, 219, 0.8)', thickness=1.5, width=8),
            mode='markers',
            marker=dict(color=colors[0], size=10, line=dict(color='rgba(0,0,0,1)', width=1)),
            name='X Component',
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.2e}<br>Std Dev: %{error_y.array:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=clean_names, 
            y=means_y, 
            error_y=dict(type='data', array=stds_y, color='rgba(46, 204, 113, 0.8)', thickness=1.5, width=8),
            mode='markers',
            marker=dict(color=colors[1], size=10, line=dict(color='rgba(0,0,0,1)', width=1)),
            name='Y Component',
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.2e}<br>Std Dev: %{error_y.array:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=clean_names, 
            y=means_z, 
            error_y=dict(type='data', array=stds_z, color='rgba(231, 76, 60, 0.8)', thickness=1.5, width=8),
            mode='markers',
            marker=dict(color=colors[2], size=10, line=dict(color='rgba(0,0,0,1)', width=1)),
            name='Z Component',
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.2e}<br>Std Dev: %{error_y.array:.2f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Force Components Mean and Std Dev - {split.capitalize()} Split",
        xaxis_title="Dataset",
        yaxis_title="Force (mean)",
        xaxis_tickangle=-45,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save the figure (PNG only)
    fig.write_image(f"stats/forces_{split}.png", scale=2)
    
    print(f"Forces plot for {split} split created in stats folder")

# 3. Stress Visualization - simplified to match energy plot style
def plot_stress(split):
    datasets = train_datasets if split == "train" else val_datasets
    
    # Extract stress data (6 components)
    stress_means = []
    stress_stds = []
    
    for dataset in datasets:
        stress_means.append(data[split][dataset]["means"]["stress"])
        stress_stds.append(data[split][dataset]["std"]["stress"])
    
    # Clean dataset names for plotting
    clean_names = [clean_dataset_name(name) for name in datasets]
    
    # Sort datasets by name for consistency
    sorted_indices = np.argsort(clean_names)
    clean_names = [clean_names[i] for i in sorted_indices]
    
    # Component names for labels
    component_names = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
    
    # Reorder stress data after sorting
    stress_means = [stress_means[i] for i in sorted_indices]
    stress_stds = [stress_stds[i] for i in sorted_indices]
    
    # Create figure for stress components
    fig = go.Figure()
    
    # Colors for the components
    colors = ['rgba(31, 119, 180, 0.7)', 'rgba(255, 127, 14, 0.7)', 'rgba(44, 160, 44, 0.7)', 
              'rgba(214, 39, 40, 0.7)', 'rgba(148, 103, 189, 0.7)', 'rgba(140, 86, 75, 0.7)']
    
    # Add traces for each stress component
    for i, comp_name in enumerate(component_names):
        # Extract the i-th component for all datasets
        component_means = [mean[i] for mean in stress_means]
        component_stds = [std[i] for std in stress_stds]
        
        fig.add_trace(
            go.Scatter(
                x=clean_names,
                y=component_means,
                error_y=dict(
                    type='data', 
                    array=component_stds, 
                    color=colors[i], 
                    thickness=1.5, 
                    width=8
                ),
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=10,
                    line=dict(color='rgba(0,0,0,1)', width=1)
                ),
                name=f'Component {comp_name}',
                hovertemplate='<b>%{x}</b><br>Mean: %{y:.4f}<br>Std Dev: %{error_y.array:.4f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Stress Components Mean and Std Dev - {split.capitalize()} Split",
        xaxis_title="Dataset",
        yaxis_title="Stress (mean)",
        xaxis_tickangle=-45,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save the figure (PNG only)
    fig.write_image(f"stats/stress_{split}.png", scale=2)
    
    print(f"Stress plot for {split} split created in stats folder")

# Generate all plots
for split in ["train", "val"]:
    plot_energy(split)
    plot_forces(split)
    plot_stress(split)

print("All visualizations completed! Files saved in the stats folder.")