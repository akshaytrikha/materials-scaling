import numpy as np
from collections import defaultdict
import pickle
from tqdm.auto import tqdm


class NaiveAtomModel:
    def __init__(self):
        # Mapping from (atom_type, neighbor_type) to sum of forces and counts
        self.force_sum = {}
        self.force_count = {}

        # Mapping from (atom_type, neighbor_type) to sum of energy contributions and counts
        self.energy_sum = {}
        self.energy_count = {}

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

                # Accumulate forces
                if atom_type not in self.force_sum.keys():
                    self.force_sum[atom_type] = {}
                    self.force_count[atom_type] = {}
                if neighbor_type not in self.force_sum[atom_type].keys():
                    self.force_sum[atom_type][neighbor_type] = 0
                    self.force_count[atom_type][neighbor_type] = 0
                self.force_sum[atom_type][neighbor_type] += forces[idx]
                self.force_count[atom_type][neighbor_type] += 1

                # Accumulate energy
                if atom_type not in self.energy_sum.keys():
                    self.energy_sum[atom_type] = {}
                    self.energy_count[atom_type] = {}
                if neighbor_type not in self.energy_sum[atom_type].keys():
                    self.energy_sum[atom_type][neighbor_type] = 0
                    self.energy_count[atom_type][neighbor_type] = 0
                self.energy_sum[atom_type][neighbor_type] += energy_per_atom
                self.energy_count[atom_type][neighbor_type] += 1

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
        overall_sum = 0
        overall_count = 0
        for atom_type in self.force_sum.keys():
            self.mean_forces[atom_type] = {}
            atom_type_sum = 0
            atom_type_count = 0
            for neighbor_type in self.force_sum[atom_type].keys():
                if self.force_count[atom_type][neighbor_type] > 0:
                    atom_type_sum += self.force_sum[atom_type][neighbor_type]
                    atom_type_count += self.force_count[atom_type][neighbor_type]
                    self.mean_forces[atom_type][neighbor_type] = self.force_sum[atom_type][neighbor_type] / self.force_count[atom_type][neighbor_type]
            if atom_type_count == 0:
                self.mean_forces[atom_type]["atom_type"] = 0
            else:
                self.mean_forces[atom_type]["atom_type"] = atom_type_sum / atom_type_count
            overall_sum += atom_type_sum
            overall_count += atom_type_count
        if overall_count == 0:
            self.mean_forces["overall"] = 0
        else:
            self.mean_forces["overall"] = overall_sum / overall_count

        self.mean_energies = {}
        overall_sum = 0
        overall_count = 0
        for atom_type in self.energy_sum.keys():
            self.mean_energies[atom_type] = {}
            atom_type_sum = 0
            atom_type_count = 0
            for neighbor_type in self.energy_sum[atom_type].keys():
                if self.energy_count[atom_type][neighbor_type] > 0:
                    atom_type_sum += self.energy_sum[atom_type][neighbor_type]
                    atom_type_count += self.energy_count[atom_type][neighbor_type]
                    self.mean_energies[atom_type][neighbor_type] = self.energy_sum[atom_type][neighbor_type] / self.energy_count[atom_type][neighbor_type]
            if atom_type_count == 0:
                self.mean_energies[atom_type]["atom_type"] = 0
            else:
                self.mean_energies[atom_type]["atom_type"] = atom_type_sum / atom_type_count
            overall_sum += atom_type_sum
            overall_count += atom_type_count
        if overall_count == 0:
            self.mean_energies["overall"] = 0
        else:
            self.mean_energies["overall"] = overall_sum / overall_count

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
            # Predict forces
            if atom_type in self.mean_forces.keys():
                if neighbor_type in self.mean_forces[atom_type].keys():
                    predicted_forces[idx] = self.mean_forces[atom_type][neighbor_type]
                else:
                    predicted_forces[idx] = self.mean_forces[atom_type]["atom_type"]
            else:
                # If unseen configuration, assign zero force or some default
                predicted_forces[idx] = self.mean_forces["overall"]

            # Predict energy
            if atom_type in self.mean_energies.keys():
                if neighbor_type in self.mean_energies[atom_type].keys():
                    predicted_energy += self.mean_energies[atom_type][neighbor_type]
                else:
                    predicted_energy += self.mean_energies[atom_type]["atom_type"]
            else:
                # If unseen configuration, assign zero force or some default
                predicted_energy += self.mean_energies["overall"]

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
