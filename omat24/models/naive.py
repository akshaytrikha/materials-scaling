from collections import defaultdict
import numpy as np
import pickle
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist


class NaiveAtomModel:
    def __init__(self, k):
        self.k = k
        # Mapping from (chain_type) to sum of forces and counts
        self.force_sum = defaultdict(float)
        self.force_count = defaultdict(int)

        # Mapping from (chain_type) to sum of energy contributions and counts
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
            distance_matrix = cdist(positions, positions, metric="euclidean")

            # Find nearest neighbors
            tuples_of_nearest_indices = self.find_nearest_neighbors_tuples(distance_matrix)

            # Estimate per-atom energy contribution
            energy_per_atom = energy / num_atoms if num_atoms > 0 else 0.0
            for idx, _ in enumerate(atoms):
                tuples_of_nearest_symbols = []
                for tuple_of_nearest_indices in tuples_of_nearest_indices[idx]:
                    tuple_of_nearest_symbols = []
                    for nearest_index in tuple_of_nearest_indices:
                        neighbor_type = symbols[nearest_index]
                        tuple_of_nearest_symbols.append(neighbor_type)
                    tuple_of_nearest_symbols = tuple(tuple_of_nearest_symbols)
                    tuples_of_nearest_symbols.append(tuple_of_nearest_symbols)
                for tuple_of_nearest_symbols in tuples_of_nearest_symbols:
                    # Accumulate forces
                    if tuple_of_nearest_symbols not in self.force_sum.keys():
                        self.force_sum[tuple_of_nearest_symbols] = 0
                        self.force_count[tuple_of_nearest_symbols] = 0
                    self.force_sum[tuple_of_nearest_symbols] += np.linalg.norm(forces[idx])
                    self.force_count[tuple_of_nearest_symbols] += 1
                    # Accumulate energy
                    if tuple_of_nearest_symbols not in self.energy_sum.keys():
                        self.energy_sum[tuple_of_nearest_symbols] = 0
                        self.energy_count[tuple_of_nearest_symbols] = 0
                    self.energy_sum[tuple_of_nearest_symbols] += energy_per_atom
                    self.energy_count[tuple_of_nearest_symbols] += 1

            # Accumulate stress
            self.stress_sum += stress
            self.stress_count += 1

    def find_nearest_neighbors_tuples(self, distance_matrix):
        """For each atom, find the index of its nearest neighbor."""
        tuples_of_nearest_indices = []
        for _, distances in enumerate(distance_matrix):
            # Use argpartition to get the top k+1 nearest indices
            nearest = np.argpartition(distances, self.k + 1)[:self.k + 1]
            nearest = nearest[np.argsort(distances[nearest])]  # Sort only the top k+1
            tuple_of_nearest_indices = [nearest[:num_nearest_neighbors].tolist() for num_nearest_neighbors in range(0, self.k + 1)]
            tuples_of_nearest_indices.append(tuple(tuple_of_nearest_indices))
        return tuples_of_nearest_indices

    def finalize(self):
        """Compute the mean forces and energy contributions after training."""
        self.mean_forces = {}
        self.mean_energies = {}
        for tuple_of_nearest_symbols in self.force_sum.keys():
            if self.force_count[tuple_of_nearest_symbols] > 0:
                self.mean_forces[tuple_of_nearest_symbols] = self.force_sum[tuple_of_nearest_symbols] / self.force_count[tuple_of_nearest_symbols]
        for tuple_of_nearest_symbols in self.energy_sum.keys():
            if self.energy_count[tuple_of_nearest_symbols] > 0:
                self.mean_energies[tuple_of_nearest_symbols] = self.energy_sum[tuple_of_nearest_symbols] / self.energy_count[tuple_of_nearest_symbols]
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
        distance_matrix = cdist(positions, positions, metric="euclidean")
        # Find nearest neighbors
        tuples_of_nearest_indices = self.find_nearest_neighbors_tuples(distance_matrix)
        # Initialize predictions
        predicted_forces = np.zeros((num_atoms))
        predicted_energies = np.zeros((num_atoms))
        for idx, _ in enumerate(atoms):
            tuples_of_nearest_symbols = []
            for tuple_of_nearest_indices in tuples_of_nearest_indices[idx]:
                tuple_of_nearest_symbols = []
                for nearest_index in tuple_of_nearest_indices:
                    neighbor_type = symbols[nearest_index]
                    tuple_of_nearest_symbols.append(neighbor_type)
                tuple_of_nearest_symbols = tuple(tuple_of_nearest_symbols)
                tuples_of_nearest_symbols.append(tuple_of_nearest_symbols)
            match_found = False
            tuple_index = len(tuples_of_nearest_symbols) - 1
            while not match_found:
                tuple_of_nearest_symbols = tuples_of_nearest_symbols[tuple_index]
                # Predict forces
                if tuple_of_nearest_symbols in self.force_sum.keys():
                        predicted_forces[idx] = self.mean_forces[tuple_of_nearest_symbols]
                        match_found = True
                tuple_index -= 1
            match_found = False
            tuple_index = len(tuples_of_nearest_symbols) - 1
            while not match_found:
                tuple_of_nearest_symbols = tuples_of_nearest_symbols[tuple_index]
                # Predict energies
                if tuple_of_nearest_symbols in self.energy_sum.keys():
                        predicted_energies[idx] = self.mean_energies[tuple_of_nearest_symbols]
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
