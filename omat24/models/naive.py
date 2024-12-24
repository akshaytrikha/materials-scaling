import numpy as np
import pickle
from collections import defaultdict
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
import torch


# Define top-level functions to replace lambda defaults
def zero_array_6():
    return np.zeros(6, dtype=np.float64)


def zero_array_3():
    return np.zeros(3, dtype=np.float64)


class NaiveMagnitudeModel:
    """Naive model that only uses force magnitudes."""

    def __init__(self, k):
        self.k = k
        self.chain_count = defaultdict(int)
        # Sums over scalars
        self.force_sum = defaultdict(float)
        self.energy_sum = defaultdict(float)
        self.stress_sum = defaultdict(zero_array_6)  # Use top-level function

    def find_nearest_neighbors_tuples(self, distance_matrix):
        tuples_of_nearest_indices = []
        for distances in distance_matrix:
            nearest = np.argpartition(distances, self.k + 1)[: self.k + 1]
            nearest = nearest[np.argsort(distances[nearest])]
            tuple_of_nearest_indices = [
                nearest[:n].tolist() for n in range(1, self.k + 2)
            ]
            tuples_of_nearest_indices.append(tuple(tuple_of_nearest_indices))
        return tuples_of_nearest_indices

    def train(self, dataset):
        for i in tqdm(range(len(dataset)), desc="Training NaiveMagnitudeModel"):
            atoms = dataset.get_atoms(i)
            symbols = atoms.get_chemical_symbols()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            positions = atoms.get_positions()
            num_atoms = len(atoms)

            distance_matrix = cdist(positions, positions, metric="euclidean")
            tuples_all = self.find_nearest_neighbors_tuples(distance_matrix)

            # Per-atom energy & stress
            e_per_atom = energy / num_atoms
            s_per_atom = stress / num_atoms

            for idx in range(num_atoms):
                # Gather neighbor symbol tuples
                for tuple_of_neighbor_inds in tuples_all[idx]:
                    tuple_of_neighbor_syms = tuple(
                        symbols[j] for j in tuple_of_neighbor_inds
                    )

                    self.chain_count[tuple_of_neighbor_syms] += 1
                    self.force_sum[tuple_of_neighbor_syms] += np.linalg.norm(
                        forces[idx]
                    )
                    self.energy_sum[tuple_of_neighbor_syms] += e_per_atom
                    self.stress_sum[tuple_of_neighbor_syms] += s_per_atom

        self.finalize()

    def finalize(self):
        self.mean_forces = {}
        self.mean_energies = {}
        self.mean_stresses = {}
        for sym_tuple in self.chain_count:
            c = self.chain_count[sym_tuple]
            self.mean_forces[sym_tuple] = self.force_sum[sym_tuple] / c
            self.mean_energies[sym_tuple] = self.energy_sum[sym_tuple] / c
            self.mean_stresses[sym_tuple] = self.stress_sum[sym_tuple] / c

    def predict_batch(self, batch_atoms):
        predicted_energy_list = []
        predicted_forces_list = []
        predicted_stress_list = []
        max_atoms = 0

        for atoms in batch_atoms:
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            num_atoms = len(atoms)
            max_atoms = max(max_atoms, num_atoms)

        for atoms in batch_atoms:
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            distance_matrix = cdist(positions, positions, metric="euclidean")
            tuples_all = self.find_nearest_neighbors_tuples(distance_matrix)
            num_atoms = len(atoms)

            predicted_energy = 0.0
            predicted_forces = np.zeros(num_atoms, dtype=np.float32)
            predicted_stress = np.zeros(6, dtype=np.float32)

            for idx in range(num_atoms):
                # Try from largest tuple to smallest
                for tuple_of_neighbor_inds in reversed(tuples_all[idx]):
                    tuple_of_neighbor_syms = tuple(
                        symbols[j] for j in tuple_of_neighbor_inds
                    )
                    if tuple_of_neighbor_syms in self.mean_forces:
                        predicted_forces[idx] = self.mean_forces[tuple_of_neighbor_syms]
                        predicted_energy += self.mean_energies[tuple_of_neighbor_syms]
                        predicted_stress += self.mean_stresses[tuple_of_neighbor_syms]
                        break

            predicted_energy_list.append(predicted_energy)
            predicted_forces_list.append(predicted_forces)
            predicted_stress_list.append(predicted_stress)

        # Pad forces
        padded_forces = []
        for forces in predicted_forces_list:
            pad_width = (0, max_atoms - len(forces))
            padded = np.pad(forces, pad_width, mode="constant")
            padded_forces.append(padded)
        return (
            torch.tensor(predicted_energy_list, dtype=torch.float32),
            torch.tensor(padded_forces, dtype=torch.float32),
            torch.tensor(predicted_stress_list, dtype=torch.float32),
        )

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


class NaiveDirectionModel:
    """Naive model uses both force magnitude and direction."""

    def __init__(self, k):
        self.k = k
        self.chain_count = defaultdict(int)
        # Now store sums of 3D vectors
        self.force_sum = defaultdict(zero_array_3)  # Use top-level function
        self.energy_sum = defaultdict(float)
        self.stress_sum = defaultdict(zero_array_6)  # Use top-level function

    def find_nearest_neighbors_tuples(self, distance_matrix):
        tuples_of_nearest_indices = []
        for distances in distance_matrix:
            nearest = np.argpartition(distances, self.k + 1)[: self.k + 1]
            nearest = nearest[np.argsort(distances[nearest])]
            tuple_of_nearest_indices = [
                nearest[:n].tolist() for n in range(1, self.k + 2)
            ]
            tuples_of_nearest_indices.append(tuple(tuple_of_nearest_indices))
        return tuples_of_nearest_indices

    def train(self, dataset):
        for i in tqdm(range(len(dataset)), desc="Training NaiveDirectionModel"):
            atoms = dataset.get_atoms(i)
            symbols = atoms.get_chemical_symbols()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()  # shape (N, 3)
            stress = atoms.get_stress()  # shape (6,)
            positions = atoms.get_positions()
            num_atoms = len(atoms)

            distance_matrix = cdist(positions, positions, metric="euclidean")
            tuples_all = self.find_nearest_neighbors_tuples(distance_matrix)

            e_per_atom = energy / num_atoms
            s_per_atom = stress / num_atoms

            for idx in range(num_atoms):
                for tuple_of_neighbor_inds in tuples_all[idx]:
                    tuple_of_neighbor_syms = tuple(
                        symbols[j] for j in tuple_of_neighbor_inds
                    )

                    self.chain_count[tuple_of_neighbor_syms] += 1
                    self.force_sum[tuple_of_neighbor_syms] += forces[
                        idx
                    ]  # Store entire vector
                    self.energy_sum[tuple_of_neighbor_syms] += e_per_atom
                    self.stress_sum[tuple_of_neighbor_syms] += s_per_atom

        self.finalize()

    def finalize(self):
        self.mean_forces = {}
        self.mean_energies = {}
        self.mean_stresses = {}
        for sym_tuple, count in self.chain_count.items():
            self.mean_forces[sym_tuple] = self.force_sum[sym_tuple] / count
            self.mean_energies[sym_tuple] = self.energy_sum[sym_tuple] / count
            self.mean_stresses[sym_tuple] = self.stress_sum[sym_tuple] / count

    def predict_batch(self, batch_atoms):
        predicted_energy_list = []
        predicted_forces_list = []
        predicted_stress_list = []
        max_atoms = 0

        # Precompute maximum number of atoms
        for atoms in batch_atoms:
            max_atoms = max(max_atoms, len(atoms))

        # Predict
        for atoms in batch_atoms:
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            distance_matrix = cdist(positions, positions, metric="euclidean")
            tuples_all = self.find_nearest_neighbors_tuples(distance_matrix)
            num_atoms = len(atoms)

            # Now we output 3D forces
            predicted_forces = np.zeros((num_atoms, 3), dtype=np.float32)
            predicted_energy = 0.0
            predicted_stress = np.zeros(6, dtype=np.float32)

            for idx in range(num_atoms):
                for tuple_of_neighbor_inds in reversed(tuples_all[idx]):
                    tuple_of_neighbor_syms = tuple(
                        symbols[j] for j in tuple_of_neighbor_inds
                    )
                    if tuple_of_neighbor_syms in self.mean_forces:
                        predicted_forces[idx, :] = self.mean_forces[
                            tuple_of_neighbor_syms
                        ]
                        predicted_energy += self.mean_energies[tuple_of_neighbor_syms]
                        predicted_stress += self.mean_stresses[tuple_of_neighbor_syms]
                        break

            predicted_energy_list.append(predicted_energy)
            predicted_forces_list.append(predicted_forces)
            predicted_stress_list.append(predicted_stress)

        # Pad forces to [batch_size, max_atoms, 3]
        padded_forces = []
        for forces in predicted_forces_list:
            N = forces.shape[0]
            pad = max_atoms - N
            if pad > 0:
                # pad along the first dimension (atoms)
                forces = np.vstack([forces, np.zeros((pad, 3), dtype=np.float32)])
            padded_forces.append(forces)

        # Convert to tensors
        predicted_energy_tensor = torch.tensor(
            predicted_energy_list, dtype=torch.float32
        )
        predicted_forces_tensor = torch.tensor(padded_forces, dtype=torch.float32)
        predicted_stress_tensor = torch.tensor(
            predicted_stress_list, dtype=torch.float32
        )

        return predicted_energy_tensor, predicted_forces_tensor, predicted_stress_tensor

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
