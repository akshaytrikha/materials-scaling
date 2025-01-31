import json
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from ase.data import chemical_symbols
import matplotlib.colors
from pathlib import Path
import imageio.v2 as imageio
import re
import argparse


def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def plot_atomic_forces(
    pl,
    subplot_idx,
    positions,
    numbers,
    forces,
    energy,
    stress,
    show_legend=False,
):
    """Plot atomic positions and forces for a single subplot."""
    pl.subplot(0, subplot_idx)

    # Create point cloud for atoms
    cloud = pv.PolyData(positions)

    # Get colormap for atomic numbers
    cmap = plt.get_cmap("tab20")
    norm = matplotlib.colors.Normalize(vmin=min(numbers), vmax=max(numbers))

    # Add atoms
    pl.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=35,
        scalars=numbers,
        cmap="tab20",
        show_scalar_bar=False,
    )

    # Add force arrows
    arrows = pv.PolyData(positions)
    arrows["vectors"] = forces
    arrows = arrows.glyph(
        orient="vectors",
        scale="vectors",
        factor=1.3,
        geom=pv.Arrow(tip_length=0.25, tip_radius=0.05, shaft_radius=0.02),
    )
    pl.add_mesh(arrows, color="red", opacity=0.5)

    # Replace Ground Truth/Prediction title with stress values
    stress_text = (
        f"σxx={stress[0]:.2f}, σyy={stress[1]:.2f}, σzz={stress[2]:.2f}\n"
        f"σyz={stress[3]:.2f}, σxz={stress[4]:.2f}, σxy={stress[5]:.2f}"
    )
    pl.add_text(
        stress_text,
        position=(0.05, 0.05),
        viewport=True,
        font_size=16,  # Reduced font size to accommodate more text
        color="black",
    )

    # Add energy value at bottom-right
    energy_text = f"{energy:.2f} eV"
    pl.add_text(
        energy_text,
        position=(0.725, 0.05),  # Moved up and adjusted for better visibility
        viewport=True,
        font_size=24,
        color="red",
    )


def plot_force_comparison(
    sample,
    epoch,
    output_path,
    model_params,
    dataset_size,
    architecture,
    lr,
    split,
):
    """Create and save a comparison plot of ground truth vs predicted forces.

    Args:
        sample: Dictionary containing the sample data
        epoch: Current epoch number
        output_path: Path to save the output image
        model_params: Number of model parameters
        dataset_size: Size of the dataset
        architecture: Model architecture name
        lr: Learning rate
    """
    # Extract data from sample
    positions = np.array(sample["positions"][0])
    numbers = np.array(sample["atomic_numbers"][0])
    forces_true = np.array(sample["true"]["forces"][0])
    forces_pred = np.array(sample["pred"]["forces"][0])
    energy_true = float(sample["true"]["energy"][0])
    energy_pred = float(sample["pred"]["energy"][0])
    stress_true = np.array(sample["true"]["stress"][0])
    stress_pred = np.array(sample["pred"]["stress"][0])

    # Create plotter
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1920, 1080])

    # Add epoch number at the top-left of the entire plot
    pl.add_text(
        f"EPOCH {epoch}",
        position=(0.05, 0.875),
        viewport=True,
        font_size=36,
        color="black",
        font="arial",
    )

    # Plot both subplots
    plot_atomic_forces(
        pl,
        0,
        positions,
        numbers,
        forces_true,
        energy_true,
        stress_true,
    )
    plot_atomic_forces(
        pl,
        1,
        positions,
        numbers,
        forces_pred,
        energy_pred,
        stress_pred,
    )

    # Set identical camera position and zoom for both views
    camera_position = [
        (25, 25, 25),  # Camera position moved further out
        (0, 0, 0),  # Focus point
        (0, 0, 1),  # Up vector
    ]

    # Apply same camera settings to both subplots
    pl.subplot(0, 0)
    pl.camera_position = camera_position
    pl.camera.zoom(0.6)  # Reduced zoom from 0.8 to 0.6

    pl.subplot(0, 1)
    pl.camera_position = camera_position
    pl.camera.zoom(0.6)  # Reduced zoom from 0.8 to 0.6

    # Add model architecture, parameters and dataset size information
    pl.add_text(
        f"arch: {architecture}",
        position=(0.71, 0.93),  # Just below epoch
        viewport=True,
        font_size=16,
        color="black",
        font="arial",
    )
    pl.add_text(
        f"#params: {model_params}",
        position=(0.71, 0.89),  # Below architecture
        viewport=True,
        font_size=16,
        color="black",
        font="arial",
    )
    pl.add_text(
        f"ds: {dataset_size}",
        position=(0.71, 0.85),  # Below model size
        viewport=True,
        font_size=16,
        color="black",
        font="arial",
    )
    pl.add_text(
        f"lr: {lr}",
        position=(0.71, 0.81),  # Below dataset size
        viewport=True,
        font_size=16,
        color="black",
        font="arial",
    )
    pl.add_text(
        f"split: {split}",
        position=(0.71, 0.77),  # Below dataset size
        viewport=True,
        font_size=16,
        color="black",
        font="arial",
    )

    # Save the image
    pl.screenshot(output_path)


def create_force_comparison_gif(
    image_dir="figures", output_name="result.gif", duration=8
):
    """Create a GIF from force comparison screenshots.

    Args:
        image_dir (str): Directory containing the images
        output_name (str): Name of the output GIF file
        duration (int): Total duration of the GIF in seconds
    """
    # Get all PNG files and sort them by epoch number
    image_files = [f for f in Path(image_dir).glob("forces_comparison_epoch_*.png")]
    image_files.sort(key=lambda x: natural_sort_key(x.name))

    if not image_files:
        print(f"No force comparison images found in {image_dir}")
        return

    # Calculate fps based on number of images and desired duration
    fps = len(image_files) / duration
    fps = min(1, fps)

    # Read all images
    images = []
    for image_file in image_files:
        images.append(imageio.imread(image_file))
        print(f"Added {image_file.name} to GIF")

    # Create the GIF
    output_path = Path(image_dir) / output_name
    imageio.mimsave(output_path, images, format="GIF", fps=fps, loop=1)
    print(f"Created GIF at {output_path} with {fps:.1f} fps ({duration}s duration)")


def main():
    """Main function to run the visualization and create GIF."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Visualize model prediction evolution."
    )
    parser.add_argument("json_file", type=str, help="Path to the JSON results file")
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of sample to visualize (default: 0)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "train"],
        default="val",
        help="Split of sample to visualize (default: val)",
    )
    args = parser.parse_args()

    # Create output directory
    Path("figures").mkdir(exist_ok=True)

    # Load data
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Get the last run
    dataset_size = list(data.keys())[-1]
    run = data[dataset_size][-1]

    # Get model configuration
    model_params = run["config"]["num_params"]
    dataset_size = run["config"]["dataset_size"]
    architecture = run["config"]["architecture"]
    lr = run["config"]["learning_rate"]

    print(
        f"Generating force comparison plots (using {args.split} split sample {args.sample_idx})..."
    )
    # Loop through all epochs
    for epoch, epoch_data in run["losses"].items():
        # Skip epochs without samples or specified sample type
        if "samples" not in epoch_data or args.split not in epoch_data["samples"]:
            continue

        # Get the specified sample for this epoch
        samples = epoch_data["samples"][args.split]
        if args.sample_idx >= len(samples):
            continue

        sample = samples[args.sample_idx]

        # Create the plot
        output_path = f"figures/forces_comparison_epoch_{epoch}.png"
        print(f"Creating plot for epoch {epoch}...")

        plot_force_comparison(
            sample=sample,
            epoch=epoch,
            output_path=output_path,
            model_params=model_params,
            dataset_size=dataset_size,
            architecture=architecture,
            lr=lr,
            split=args.split,
        )

    print("Finished creating all plots.")

    print("\nGenerating GIF from plots...")
    create_force_comparison_gif(duration=8)
    print("Done!")


if __name__ == "__main__":
    main()
