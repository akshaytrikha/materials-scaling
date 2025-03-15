# External
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import os
import sys

# Internal
from data import OMat24Dataset
from models.transformer_models import XTransformerModel


class SimpleTransformerAnalyzer:
    """A simplified analyzer for transformer models trained on atomic structures."""

    def __init__(self, model_path, device="cuda"):
        """
        Initialize with a trained model

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        # Load the model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Register hooks to capture attention
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()

        print(f"Model loaded and hooks registered")

    def _load_model(self, model_path):
        """Load the transformer model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = {"d_model": 160, "depth": 3, "n_heads": 2, "d_ff_mult": 4}

        # Try to extract config from checkpoint if available
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
        elif "config" in checkpoint:
            config = checkpoint["config"]

        model = XTransformerModel(
            num_tokens=119,  # Max atomic number in periodic table
            d_model=config.get("d_model", 160),
            depth=config.get("depth", 3),
            n_heads=config.get("n_heads", 2),
            d_ff_mult=config.get("d_ff_mult", 4),
            use_factorized=config.get("use_factorized", False),
        )

        # Load the state dictionary
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Try loading the checkpoint directly as a state dict
            model.load_state_dict(checkpoint)

        return model

    def _register_hooks(self):
        """Register hooks to directly capture attention maps from transformer layers using the exact model structure"""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []

        # IMPORTANT: The exact path to attention in this specific model structure
        attention_layers = [
            (1, "Attention Layer 0"),  # Index 1 in the ModuleList = first attention
            (4, "Attention Layer 1"),  # Index 4 = second attention
            (7, "Attention Layer 2"),  # Index 7 = third attention
        ]

        for layer_idx, layer_name in attention_layers:
            # Get the attend module directly using the model structure we saw in the debug output
            try:
                # This is the direct path to the Attend module based on the model printout
                attend_module = self.model.attn_layers.layers[layer_idx][1].attend

                # Create a hook function for this layer
                def make_hook(layer_name):
                    def hook(module, inputs, outputs):
                        # Directly capture and store the raw attention weights
                        # Get query and key from inputs
                        q, k, v = inputs[0], inputs[1], inputs[2]

                        # Calculate attention scores manually
                        # Compute QK^T
                        attn_weights = torch.matmul(q, k.transpose(-2, -1))

                        # Scale by sqrt(d_k)
                        d_k = q.size(-1)
                        attn_weights = attn_weights / (d_k**0.5)

                        # Apply softmax
                        attn_weights = torch.softmax(attn_weights, dim=-1)

                        self.attention_maps.append(
                            (f"layers.{layer_idx//3}.attention", attn_weights.detach())
                        )
                        print(
                            f"Captured attention map for {layer_name}, shape: {attn_weights.shape}"
                        )

                    return hook

                # Register the hook to the attend module
                hook = attend_module.register_forward_hook(make_hook(layer_name))
                self.hooks.append(hook)
                print(f"Successfully registered hook for {layer_name}")

            except (AttributeError, IndexError) as e:
                print(f"Failed to register hook for {layer_name}: {e}")

        if not self.hooks:
            print(
                "WARNING: No hooks were successfully registered! Attention visualization will not work."
            )

    def analyze_sample(self, atoms_data):
        """
        Run inference on a sample and collect attention patterns

        Args:
            atoms_data: Dictionary with input data (atomic_numbers, positions, etc.)

        Returns:
            Dictionary of model outputs and attention patterns
        """
        # Clear previous attention data
        self.attention_maps = []

        # Prepare inputs for the model
        atomic_numbers = atoms_data["atomic_numbers"].unsqueeze(0).to(self.device)
        positions = atoms_data["positions"].unsqueeze(0).to(self.device)

        # Create mask for padding
        mask = torch.ones_like(atomic_numbers).bool().to(self.device)

        # Get distance matrix if needed
        if "distance_matrix" in atoms_data:
            distance_matrix = atoms_data["distance_matrix"].unsqueeze(0).to(self.device)
        else:
            # Calculate distance matrix if not provided
            distance_matrix = self._calculate_distance_matrix(positions)

        # Run the model in eval mode
        with torch.no_grad():
            forces, energy, stress = self.model(
                atomic_numbers, positions, distance_matrix, mask
            )

        # Return results
        results = {
            "forces": forces.cpu(),
            "energy": energy.cpu(),
            "stress": stress.cpu(),
            "attention_maps": [
                (name, attn.cpu()) for name, attn in self.attention_maps
            ],
        }

        return results

    def _calculate_distance_matrix(self, positions):
        """Calculate distance matrix from positions"""
        n_batch, n_atoms, _ = positions.shape

        # Reshape for broadcasting
        pos_i = positions.unsqueeze(2)  # [batch, n_atoms, 1, 3]
        pos_j = positions.unsqueeze(1)  # [batch, 1, n_atoms, 3]

        # Calculate pairwise distances
        dist = torch.sqrt(
            torch.sum((pos_i - pos_j) ** 2, dim=-1) + 1e-8
        )  # [batch, n_atoms, n_atoms]

        return dist

    def visualize_atom_attention(self, atoms_data, layer_idx=0, head_idx=0):
        """
        Visualize atom-to-atom attention for a specific layer and head

        Args:
            atoms_data: Dictionary with input data
            layer_idx: Index of the transformer layer to visualize (0, 1, or 2)
            head_idx: Index of the attention head to visualize

        Returns:
            Plotly figure object
        """
        # Run analysis to get attention maps
        results = self.analyze_sample(atoms_data)

        # Extract atomic numbers for labels
        atomic_numbers = atoms_data["atomic_numbers"]
        n_atoms = len(atomic_numbers)

        # Get element names from atomic numbers
        element_map = {
            1: "H",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            14: "Si",
            15: "P",
            16: "S",
            17: "Cl",
            26: "Fe",
            29: "Cu",
            30: "Zn",
        }

        elements = [
            f"{i}:{element_map.get(num.item(), str(num.item()))}"
            for i, num in enumerate(atomic_numbers)
        ]

        # Find the attention map for the specified layer
        target_name = f"layers.{layer_idx}.attention"
        target_attn = None

        for name, attn in results["attention_maps"]:
            if name == target_name:
                # Extract attention pattern for the specified head
                if attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    target_attn = attn[0, head_idx].numpy()
                    print(f"Found attention map from: {name}, shape: {attn.shape}")
                    break

        # If no attention map found, check if we have any attention maps
        if target_attn is None:
            print(
                f"Could not find attention map for layer {layer_idx}, head {head_idx}"
            )
            if results["attention_maps"]:
                # Try to use any available attention map
                print("Available attention maps:")
                for name, attn in results["attention_maps"]:
                    print(f"  - {name}: shape {attn.shape}")

                # Use the first available attention map
                name, attn = results["attention_maps"][0]
                print(f"Using attention map from: {name}")

                if attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    target_attn = attn[0, min(head_idx, attn.size(1) - 1)].numpy()
                else:
                    print(f"Unexpected attention map shape: {attn.shape}")
                    # Create a dummy attention
                    target_attn = np.eye(n_atoms)
            else:
                # If no attention maps found, create a dummy
                print("No attention maps found. Creating placeholder visualization.")
                target_attn = np.eye(n_atoms)  # Identity matrix as placeholder

        # Create heatmap visualization
        fig = go.Figure(
            data=go.Heatmap(
                z=target_attn,
                x=elements,
                y=elements,
                colorscale="Viridis",
                colorbar=dict(title="Attention Weight"),
                text=[
                    [f"From {i} to {j}: {val:.3f}" for j, val in enumerate(row)]
                    for i, row in enumerate(target_attn)
                ],
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Atom-to-Atom Attention (Layer {layer_idx}, Head {head_idx})",
            xaxis=dict(title="Target Atoms"),
            yaxis=dict(title="Source Atoms"),
            width=800,
            height=700,
        )

        return fig

    def analyze_distance_vs_attention(self, atoms_data, layer_idx=0, head_idx=0):
        """
        Analyze how attention correlates with interatomic distances

        Args:
            atoms_data: Dictionary with input data
            layer_idx: Index of the transformer layer to analyze (0, 1, or 2)
            head_idx: Index of the attention head to analyze

        Returns:
            Matplotlib figure with scatter plot
        """
        # Calculate distance matrix if not provided
        if "distance_matrix" not in atoms_data:
            positions = atoms_data["positions"]
            n_atoms = positions.shape[0]
            distance_matrix = torch.zeros((n_atoms, n_atoms))
            for i in range(n_atoms):
                for j in range(n_atoms):
                    distance_matrix[i, j] = torch.norm(positions[i] - positions[j])
        else:
            distance_matrix = atoms_data["distance_matrix"]

        # Get attention maps
        results = self.analyze_sample(atoms_data)

        # Find the attention map for the specified layer
        target_name = f"layers.{layer_idx}.attention"
        target_attn = None

        for name, attn in results["attention_maps"]:
            if name == target_name:
                # Extract attention pattern for the specified head
                if attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    target_attn = attn[0, head_idx]
                    print(f"Found attention map from: {name}, shape: {attn.shape}")
                    break

        # If no specific attention map found, try any available map
        if target_attn is None:
            print(
                f"Could not find attention map for layer {layer_idx}, head {head_idx}"
            )
            if results["attention_maps"]:
                # Print available maps
                print("Available attention maps:")
                for name, attn in results["attention_maps"]:
                    print(f"  - {name}: shape {attn.shape}")

                # Use the first available map
                name, attn = results["attention_maps"][0]
                print(f"Using attention map from: {name}")

                if attn.dim() == 4:
                    target_attn = attn[
                        0, min(head_idx, attn.size(1) - 1)
                    ]  # First batch, specified head
                elif attn.dim() == 3:
                    target_attn = attn[
                        min(head_idx, attn.size(0) - 1)
                    ]  # Specified head
                else:
                    target_attn = attn
            else:
                print("No attention maps found.")
                return None

        # Convert to numpy for plotting
        distance_matrix = distance_matrix.numpy()
        attention_matrix = target_attn.numpy()

        # Collect pairs of distance and attention (excluding self-attention)
        distances = []
        attentions = []
        n_atoms = distance_matrix.shape[0]

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:  # Skip self-attention
                    distances.append(distance_matrix[i, j])
                    attentions.append(attention_matrix[i, j])

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(distances, attentions, alpha=0.6)

        # Add trendline
        z = np.polyfit(distances, attentions, 1)
        p = np.poly1d(z)
        ax.plot(sorted(distances), p(sorted(distances)), "r--", alpha=0.8)

        # Compute correlation
        corr = np.corrcoef(distances, attentions)[0, 1]

        # Add labels and title
        ax.set_xlabel("Interatomic Distance (Å)")
        ax.set_ylabel("Attention Weight")
        ax.set_title(
            f"Distance vs. Attention (Layer {layer_idx}, Head {head_idx})\nCorrelation: {corr:.3f}"
        )
        ax.grid(True, alpha=0.3)

        return fig

    def save_figures(self, atoms_data, output_dir="output_figures"):
        """
        Generate and save both visualization figures to the specified directory

        Args:
            atoms_data: Dictionary with input data
            output_dir: Directory to save figures (will be created if it doesn't exist)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate and save attention heatmap
        print("Generating atom attention heatmap...")
        heatmap_fig = self.visualize_atom_attention(atoms_data)
        if heatmap_fig:
            # Save as HTML for interactive visualization
            heatmap_path = os.path.join(output_dir, "atom_attention_heatmap.html")
            heatmap_fig.write_html(heatmap_path)
            print(f"Saved attention heatmap to {heatmap_path}")

            # Try to save as image, but handle potential Kaleido errors
            img_path = os.path.join(output_dir, "atom_attention_heatmap.png")
            try:
                heatmap_fig.write_image(img_path)
                print(f"Saved attention heatmap image to {img_path}")
            except Exception as e:
                print(f"Warning: Failed to save PNG image: {str(e)}")
                print(
                    "The HTML version was saved successfully and can be viewed in any browser"
                )

        # Generate and save distance vs attention plot
        print("\nGenerating distance vs attention plot...")
        scatter_fig = self.analyze_distance_vs_attention(atoms_data)
        if scatter_fig:
            # Save matplotlib figure
            scatter_path = os.path.join(output_dir, "distance_vs_attention.png")
            scatter_fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close(scatter_fig)  # Close the figure to free memory
            print(f"Saved distance vs attention plot to {scatter_path}")

        print(f"\nAll visualizations complete and saved to '{output_dir}' directory")

        return {
            "heatmap_path": os.path.join(output_dir, "atom_attention_heatmap.html"),
            "scatter_path": os.path.join(output_dir, "distance_vs_attention.png"),
        }


if __name__ == "__main__":
    output_dir = "interp_results"

    # Initialize the analyzer with your model checkpoint
    print(f"Initializing SimpleTransformerAnalyzer...")
    analyzer = SimpleTransformerAnalyzer(
        "checkpoints/Transformer_ds1020_p1019086_20250314_112836.pth"
    )

    # Load a sample from your dataset
    print(f"Loading dataset sample...")
    dataset = OMat24Dataset(
        [Path("datasets/val/rattled-300-subsampled")], architecture="Transformer"
    )
    sample = dataset[0]

    # Save both figures to the output directory
    print(f"Saving figures to '{output_dir}'...")
    output_paths = analyzer.save_figures(sample, output_dir=output_dir)

    print("Done!")
