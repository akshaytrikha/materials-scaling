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
            tuples_of_nearest_indices = self.find_nearest_neighbors(distance_matrix)

            # Estimate per-atom energy contribution
            energy_per_atom = energy / num_atoms if num_atoms > 0 else 0.0
            for idx, _ in enumerate(atoms):
                atom_type = symbols[idx]
                tuples_of_nearest_symbols = []
                for tuple_of_nearest_indices in tuples_of_nearest_indices[idx]:
                    tuple_of_nearest_symbols = []
                    for nearest_index in tuple_of_nearest_indices:
                        neighbor_type = symbols[nearest_index]
                        tuple_of_nearest_symbols.append(neighbor_type)
                    tuple_of_nearest_symbols = tuple(tuple_of_nearest_symbols)
                for tuple_of_nearest_symbols in tuples_of_nearest_symbols:
                    # Accumulate forces
                    if atom_type not in self.force_sum.keys():
                        self.force_sum[atom_type] = {}
                        self.force_count[atom_type] = {}
                    if tuple_of_nearest_symbols not in self.force_sum[atom_type].keys():
                        self.force_sum[atom_type][tuple_of_nearest_symbols] = 0
                        self.force_count[atom_type][tuple_of_nearest_symbols] = 0
                    self.force_sum[atom_type][tuple_of_nearest_symbols] += np.linalg.norm(forces[idx])
                    self.force_count[atom_type][tuple_of_nearest_symbols] += 1
                    # Accumulate energy
                    if atom_type not in self.energy_sum.keys():
                        self.energy_sum[atom_type] = {}
                        self.energy_count[atom_type] = {}
                    if tuple_of_nearest_symbols not in self.energy_sum[atom_type].keys():
                        self.energy_sum[atom_type][tuple_of_nearest_symbols] = 0
                        self.energy_count[atom_type][tuple_of_nearest_symbols] = 0
                    self.energy_sum[atom_type][tuple_of_nearest_symbols] += energy_per_atom
                    self.energy_count[atom_type][tuple_of_nearest_symbols] += 1

            # Accumulate stress
            self.stress_sum += stress
            self.stress_count += 1

    def find_nearest_neighbors_tuples(self, distance_matrix, k):
        """For each atom, find the index of its nearest neighbor."""
        tuples_of_nearest_indices = []
        for num_nearest_neighbors in range(k + 1):
            tuple_of_nearest_indices = ()
            for i, distances in enumerate(distance_matrix):
                # Exclude self-distance by setting it to infinity
                distances = distances.copy()
                distances[i] = np.inf
                # Get indices of the k smallest distances
                nearest = np.argsort(distances)[:num_nearest_neighbors]
                tuple_of_nearest_indices.append(nearest.tolist())
            tuples_of_nearest_indices.append(tuple_of_nearest_indices)
        return tuples_of_nearest_indices

    def finalize(self):
        """Compute the mean forces and energy contributions after training."""
        self.mean_forces = {}
        for atom_type in self.force_sum.keys():
            self.mean_forces[atom_type] = {}
            for tuple_of_nearest_symbols in self.force_sum[atom_type].keys():
                if self.force_count[atom_type][tuple_of_nearest_symbols] > 0:
                    self.mean_forces[atom_type][tuple_of_nearest_symbols] = self.force_sum[atom_type][tuple_of_nearest_symbols] / self.force_count[atom_type][tuple_of_nearest_symbols]
        for atom_type in self.energy_sum.keys():
            self.mean_energies[atom_type] = {}
            for tuple_of_nearest_symbols in self.energy_sum[atom_type].keys():
                if self.energy_count[atom_type][tuple_of_nearest_symbols] > 0:
                    self.mean_energies[atom_type][tuple_of_nearest_symbols] = self.energy_sum[atom_type][tuple_of_nearest_symbols] / self.energy_count[atom_type][tuple_of_nearest_symbols]
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
            return 0.0, 0.0, self.mean_stress
        # Compute distance matrix
        distance_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        # Find nearest neighbors
        tuples_of_nearest_indices = self.find_nearest_neighbors(distance_matrix)
        # Initialize predictions
        predicted_forces = np.zeros((num_atoms))
        predicted_energies = np.zeros((num_atoms))
        for idx, _ in enumerate(atoms):
            atom_type = symbols[idx]
            tuples_of_nearest_symbols = []
            for tuple_of_nearest_indices in tuples_of_nearest_indices[idx]:
                tuple_of_nearest_symbols = []
                for nearest_index in tuple_of_nearest_indices:
                    neighbor_type = symbols[nearest_index]
                    tuple_of_nearest_symbols.append(neighbor_type)
                tuple_of_nearest_symbols = tuple(tuple_of_nearest_symbols)
            match_found = False
            tuple_index = len(tuples_of_nearest_symbols) - 1
            while not match_found:
                tuple_of_nearest_symbols = tuples_of_nearest_symbols[tuple_index]
                # Predict forces
                if atom_type in self.mean_forces.keys() and tuple_of_nearest_symbols in self.force_sum[atom_type].keys():
                        predicted_forces[idx] = self.mean_forces[atom_type][tuple_of_nearest_symbols]
                        match_found = True
                tuple_index -= 1
            while not match_found:
                tuple_of_nearest_symbols = tuples_of_nearest_symbols[tuple_index]
                # Predict energies
                if atom_type in self.mean_forces.keys() and tuple_of_nearest_symbols in self.energy_sum[atom_type].keys():
                        predicted_energies[idx] = self.mean_energies[atom_type][tuple_of_nearest_symbols]
                        match_found = True
                tuple_index -= 1

        # Predict stress as the mean stress from training
        predicted_stress = self.mean_stress.copy()

        return predicted_energies, predicted_forces, predicted_stress

    def save(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
