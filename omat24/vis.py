# External
from pathlib import Path
import random
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from ase.data import chemical_symbols
import matplotlib.colors

# Internal
from data import OMat24Dataset, download_dataset


dataset_name = "val-rattled-300-subsampled"
dataset_path = Path(f"datasets/{dataset_name}")
if not dataset_path.exists():
    download_dataset(dataset_name)

dataset = OMat24Dataset(dataset_path=dataset_path)
sample = random.choice(dataset)

positions = sample["positions"].numpy()
numbers = sample["atomic_numbers"].numpy()
forces = sample["forces"].numpy()
print("Atomic numbers:", numbers)
print("Positions:", positions)
print("Forces:", forces)

# Create a mapping of atomic numbers to element symbols
unique_numbers = np.unique(numbers)
element_labels = [f"{num} ({chemical_symbols[num]})" for num in unique_numbers]

# Create PyVista point cloud
cloud = pv.PolyData(positions)

# Create a plotter
pl = pv.Plotter()

# Get the colormap
cmap = plt.get_cmap("tab20")
norm = matplotlib.colors.Normalize(vmin=min(numbers), vmax=max(numbers))

# Add the atoms as spheres
pl.add_mesh(
    cloud,
    render_points_as_spheres=True,
    point_size=35,
    scalars=numbers,
    cmap="tab20",
    show_scalar_bar=False,
)

# Add force arrows
scale_factor = 1.3  # Adjust this value to make arrows longer/shorter
start_points = positions

# Create arrows with proper orientation
arrows = pv.PolyData(start_points)
arrows["vectors"] = forces  # Add force vectors as a data array

# Create glyphs with proper orientation
arrows = arrows.glyph(
    orient="vectors",  # Use the vectors array for orientation
    scale="vectors",  # Scale arrows by force magnitude
    factor=scale_factor,  # Overall scaling factor
    geom=pv.Arrow(tip_length=0.25, tip_radius=0.05, shaft_radius=0.02),
)

# Add arrows to the plot
pl.add_mesh(
    arrows,
    color="red",
    opacity=0.5,
)

# Add text annotations for each element
for i, num in enumerate(unique_numbers):
    color = cmap(norm(num))

    # Add the colored label
    pl.add_text(
        "•",
        position=(0.01, 0.94 - i * 0.05, 0),
        viewport=True,
        font_size=22,
        color=color[:3],
    )

    # Add the text label
    pl.add_text(
        f"{num} ({chemical_symbols[num]})",
        position=(0.04, 0.95 - i * 0.05, 0),
        viewport=True,
        font_size=12,
        color="black",
    )

# Set camera position for better view
pl.camera_position = "xy"
pl.camera.zoom(0.9)

# Show the plot
pl.show()
