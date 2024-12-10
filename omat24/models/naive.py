from collections import defaultdict
import numpy as np
import pickle
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
import torch

class NaiveAtomModel:
    def __init__(self, k):
        self.k = k
        self.chain_count = defaultdict(int)
        # Mapping from (chain_type) to sum of forces and counts
        self.force_sum = defaultdict(float)

        # Mapping from (chain_type) to sum of energy contributions and counts
        self.energy_sum = defaultdict(float)

        # For stress, we'll store sum and count separately
        self.stress_sum = defaultdict(float)

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
            energy_per_atom = energy / num_atoms
            stress_per_atom = tuple(stress / num_atoms)
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
                        self.chain_count[tuple_of_nearest_symbols] = 0
                        self.force_sum[tuple_of_nearest_symbols] = 0
                        self.energy_sum[tuple_of_nearest_symbols] = 0
                        self.stress_sum[tuple_of_nearest_symbols] = (0, 0, 0, 0, 0, 0)
                    self.chain_count[tuple_of_nearest_symbols] += 1
                    self.force_sum[tuple_of_nearest_symbols] += np.linalg.norm(forces[idx])
                    self.energy_sum[tuple_of_nearest_symbols] += energy_per_atom
                    self.stress_sum[tuple_of_nearest_symbols] = tuple(a + b for a, b in zip(self.stress_sum[tuple_of_nearest_symbols], stress_per_atom))

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
        self.mean_stresses = {}
        for tuple_of_nearest_symbols in self.force_sum.keys():
            if self.chain_count[tuple_of_nearest_symbols] > 0:
                self.mean_forces[tuple_of_nearest_symbols] = self.force_sum[tuple_of_nearest_symbols] / self.chain_count[tuple_of_nearest_symbols]
                self.mean_energies[tuple_of_nearest_symbols] = self.energy_sum[tuple_of_nearest_symbols] / self.chain_count[tuple_of_nearest_symbols]
                self.mean_stresses[tuple_of_nearest_symbols] = tuple(x / self.chain_count[tuple_of_nearest_symbols] for x in self.stress_sum[tuple_of_nearest_symbols])

    def predict_batch(self, batch_atoms):
        """Predict the forces, energy, and stress for a given atoms object."""
        symbols_list = [atoms.get_chemical_symbols() for atoms in batch_atoms]
        positions_list = [atoms.get_positions() for atoms in batch_atoms]
        num_atoms_list = [len(atoms) for atoms in batch_atoms]
        distance_matrices = [cdist(positions, positions, metric="euclidean") for positions in positions_list]
        # Initialize predictions
        predicted_energy_results = []
        predicted_forces_results = []
        predicted_stress_results = []
        for atoms, symbols, num_atoms, distance_matrix in zip(batch_atoms, symbols_list, num_atoms_list, distance_matrices):
            # Find nearest neighbors
            tuples_of_nearest_indices = self.find_nearest_neighbors_tuples(distance_matrix)
            predicted_forces = np.zeros((num_atoms))
            predicted_energy = 0
            predicted_stress = (0, 0, 0, 0, 0, 0)
            for idx in range(len(atoms)):
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
                    # Predict results
                    if tuple_of_nearest_symbols in self.force_sum.keys():
                            predicted_forces[idx] = self.mean_forces[tuple_of_nearest_symbols]
                            predicted_energy += self.mean_energies[tuple_of_nearest_symbols]
                            predicted_stress = tuple(a + b for a, b in zip(predicted_stress, self.mean_stresses[tuple_of_nearest_symbols]))
                            match_found = True
                    tuple_index -= 1
                match_found = False
            predicted_energy_results.append(predicted_energy)
            predicted_forces_results.append(predicted_forces)
            predicted_stress_results.append(predicted_stress)
        predicted_energy_results = torch.tensor(predicted_energy_results)
        max_atoms = max(force.shape[0] for force in predicted_forces_results)
        # Pad forces to have the same number of atoms
        padded_forces = []
        for force in predicted_forces_results:
            pad_width = (0, max_atoms - force.shape[0])  # Pad along atom dimension
            padded_forces.append(np.pad(force, pad_width, mode='constant'))
        # Convert to a PyTorch tensor
        predicted_forces_results = torch.tensor(np.array(padded_forces))
        predicted_stress_results = torch.tensor(predicted_stress_results)
        return predicted_energy_results, predicted_forces_results, predicted_stress_results

    def save(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
