import numpy as np
from collections import defaultdict
import pickle
from tqdm.auto import tqdm


class NaiveAtomModel:
    def __init__(self):
        # Mapping from (atom_type, neighbor_type) to sum of forces and counts
        self.force_sum = defaultdict(self.zero_vector)
        self.force_count = defaultdict(int)

        # Mapping from (atom_type, neighbor_type) to sum of energy contributions and counts
        self.energy_sum = defaultdict(float)
        self.energy_count = defaultdict(int)

        # For stress, we'll store sum and count separately
        self.stress_sum = np.zeros(6)
        self.stress_count = 0

    @staticmethod
    def zero_vector():
        return np.zeros(3)

    def train(self, dataset):
        """Train the model by iterating over the dataset and accumulating sums and counts."""
        for i in tqdm(range(len(dataset)), desc="Training Model"):
            atoms = dataset.get_atoms(i)
            symbols = atoms.get_chemical_symbols()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            positions = atoms.get_positions()
            num_atoms = len(atoms)

            # Compute distance matrix
            distance_matrix = np.linalg.norm(
                positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
            )

            # Find nearest neighbors
            nearest_indices = self.find_nearest_neighbors(distance_matrix)

            # Estimate per-atom energy contribution
            energy_per_atom = energy / num_atoms if num_atoms > 0 else 0.0

            for idx, atom in enumerate(atoms):
                atom_type = symbols[idx]
                neighbor_idx = nearest_indices[idx]
                neighbor_type = symbols[neighbor_idx]

                key = (atom_type, neighbor_type)

                # Accumulate forces
                self.force_sum[key] += forces[idx]
                self.force_count[key] += 1

                # Accumulate energy
                self.energy_sum[key] += energy_per_atom
                self.energy_count[key] += 1

            # Accumulate stress
            self.stress_sum += stress
            self.stress_count += 1

    def find_nearest_neighbors(self, distance_matrix):
        """For each atom, find the index of its nearest neighbor."""
        nearest_indices = []
        for i, distances in enumerate(distance_matrix):
            # Exclude self-distance by setting it to infinity
            distances = distances.copy()
            distances[i] = np.inf
            nearest = np.argmin(distances)
            nearest_indices.append(nearest)
        return nearest_indices

    def finalize(self):
        """Compute the mean forces and energy contributions after training."""
        self.mean_forces = {}
        for key in self.force_sum:
            if self.force_count[key] > 0:
                self.mean_forces[key] = self.force_sum[key] / self.force_count[key]

        self.mean_energy = {}
        for key in self.energy_sum:
            if self.energy_count[key] > 0:
                self.mean_energy[key] = self.energy_sum[key] / self.energy_count[key]

        # Compute mean stress
        self.mean_stress = (
            self.stress_sum / self.stress_count
            if self.stress_count > 0
            else np.zeros(6)
        )

    def predict(self, atoms):
        """Predict the forces, energy, and stress for a given atoms object."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        num_atoms = len(atoms)

        if num_atoms == 0:
            return 0.0, np.zeros((0, 3)), self.mean_stress

        # Compute distance matrix
        distance_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )

        # Find nearest neighbors
        nearest_indices = self.find_nearest_neighbors(distance_matrix)

        # Initialize predictions
        predicted_forces = np.zeros((num_atoms, 3))
        predicted_energy = 0.0

        for idx, atom in enumerate(atoms):
            atom_type = symbols[idx]
            neighbor_idx = nearest_indices[idx]
            neighbor_type = symbols[neighbor_idx]

            key = (atom_type, neighbor_type)

            # Predict forces
            if key in self.mean_forces:
                predicted_forces[idx] = self.mean_forces[key]
            else:
                # If unseen configuration, assign zero force or some default
                predicted_forces[idx] = np.zeros(3)

            # Predict energy
            if key in self.mean_energy:
                predicted_energy += self.mean_energy[key]
            else:
                # If unseen configuration, assign zero energy contribution or some default
                predicted_energy += 0.0

        # Predict stress as the mean stress from training
        predicted_stress = self.mean_stress.copy()

        return predicted_energy, predicted_forces, predicted_stress

    def save(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
