import numpy as np
import pickle
from collections import defaultdict
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable


# Utility functions to init ndarrays since lambda fn can't be pickled)
def zero_array_6() -> np.ndarray:
    return np.zeros(6, dtype=np.float64)


def zero_array_3() -> np.ndarray:
    return np.zeros(3, dtype=np.float64)


class NaiveModel(ABC):
    """
    Abstract Base Class for Naive Models.
    Defines the interface and shared methods for all naive models.
    """

    @abstractmethod
    def train(self, dataset: Any) -> None:
        """Train the model using the provided dataset."""
        pass

    @abstractmethod
    def predict_batch(
        self, batch_atoms: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict energies, forces, and stresses for a batch of structures."""
        pass

    def save(self, filepath: str) -> None:
        """Save the trained model to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "NaiveModel":
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def find_nearest_neighbors_tuples(
        self, distance_matrix: np.ndarray
    ) -> List[Tuple[List[int], ...]]:
        """
        Find tuples of nearest neighbor indices for each atom in the distance matrix.

        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix.

        Returns:
            List[Tuple[List[int], ...]]: List of tuples containing neighbor indices.
        """
        tuples_of_nearest_indices = []
        for distances in distance_matrix:
            nearest = np.argpartition(distances, self.k + 1)[: self.k + 1]
            nearest = nearest[np.argsort(distances[nearest])]
            tuple_of_nearest_indices = [
                nearest[:n].tolist() for n in range(1, self.k + 2)
            ]
            tuples_of_nearest_indices.append(tuple(tuple_of_nearest_indices))
        return tuples_of_nearest_indices

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of nearest neighbors to consider."""
        pass


class NaiveMagnitudeModel(NaiveModel):
    """Naive model that only uses force magnitudes."""

    def __init__(self, k: int):
        self._k = k
        self.chain_count: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.force_sum: Dict[Tuple[str, ...], float] = defaultdict(float)
        self.energy_sum: Dict[Tuple[str, ...], float] = defaultdict(float)
        self.stress_sum: Dict[Tuple[str, ...], np.ndarray] = defaultdict(zero_array_6)
        # Mean values after training
        self.mean_forces: Dict[Tuple[str, ...], float] = {}
        self.mean_energies: Dict[Tuple[str, ...], float] = {}
        self.mean_stresses: Dict[Tuple[str, ...], np.ndarray] = {}

    @property
    def k(self) -> int:
        return self._k

    def train(self, dataset: Any) -> None:
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

    def finalize(self) -> None:
        self.mean_forces = {}
        self.mean_energies = {}
        self.mean_stresses = {}
        for sym_tuple, count in self.chain_count.items():
            self.mean_forces[sym_tuple] = self.force_sum[sym_tuple] / count
            self.mean_energies[sym_tuple] = self.energy_sum[sym_tuple] / count
            self.mean_stresses[sym_tuple] = self.stress_sum[sym_tuple] / count

    def predict_batch(
        self, batch_atoms: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_energy_list: List[float] = []
        predicted_forces_list: List[np.ndarray] = []
        predicted_stress_list: List[np.ndarray] = []
        max_atoms = 0

        # Determine the maximum number of atoms in the batch for padding
        for atoms in batch_atoms:
            num_atoms = len(atoms)
            if num_atoms > max_atoms:
                max_atoms = num_atoms

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


class NaiveDirectionModel(NaiveModel):
    """Naive model that uses both force magnitude and direction."""

    def __init__(self, k: int):
        self._k = k
        self.chain_count: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.force_sum: Dict[Tuple[str, ...], np.ndarray] = defaultdict(zero_array_3)
        self.energy_sum: Dict[Tuple[str, ...], float] = defaultdict(float)
        self.stress_sum: Dict[Tuple[str, ...], np.ndarray] = defaultdict(zero_array_6)
        # Mean values after training
        self.mean_forces: Dict[Tuple[str, ...], np.ndarray] = {}
        self.mean_energies: Dict[Tuple[str, ...], float] = {}
        self.mean_stresses: Dict[Tuple[str, ...], np.ndarray] = {}

    @property
    def k(self) -> int:
        return self._k

    def train(self, dataset: Any) -> None:
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
                    self.force_sum[tuple_of_neighbor_syms] += forces[idx]
                    self.energy_sum[tuple_of_neighbor_syms] += e_per_atom
                    self.stress_sum[tuple_of_neighbor_syms] += s_per_atom

        self.finalize()

    def finalize(self) -> None:
        self.mean_forces = {}
        self.mean_energies = {}
        self.mean_stresses = {}
        for sym_tuple, count in self.chain_count.items():
            self.mean_forces[sym_tuple] = self.force_sum[sym_tuple] / count
            self.mean_energies[sym_tuple] = self.energy_sum[sym_tuple] / count
            self.mean_stresses[sym_tuple] = self.stress_sum[sym_tuple] / count

    def predict_batch(
        self, batch_atoms: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_energy_list: List[float] = []
        predicted_forces_list: List[np.ndarray] = []
        predicted_stress_list: List[np.ndarray] = []
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
            pad = max_atoms - forces.shape[0]
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


class NaiveMeanModel(NaiveModel):
    """Naive model that predicts the overall mean of energies, forces, and stresses."""

    def __init__(self):
        # Initialize accumulators
        self.total_energy: float = 0.0
        self.total_force: np.ndarray = np.zeros(3, dtype=np.float64)
        self.total_stress: np.ndarray = np.zeros(6, dtype=np.float64)
        self.total_atoms: int = 0
        self.total_structures: int = 0

        # Mean values (computed after training)
        self.mean_energy_per_atom: float = 0.0
        self.mean_force: np.ndarray = np.zeros(3, dtype=np.float64)
        self.mean_stress: np.ndarray = np.zeros(6, dtype=np.float64)

    @property
    def k(self) -> int:
        # Not applicable for NaiveMeanModel
        return 0

    def train(self, dataset: Any) -> None:
        """Train the model by computing the mean energy, force, and stress over the dataset."""
        for i in tqdm(range(len(dataset)), desc="Training NaiveMeanModel"):
            atoms = dataset.get_atoms(i)
            energy = atoms.get_potential_energy()  # Total energy for the structure
            forces = atoms.get_forces()  # Forces on each atom, shape (N, 3)
            stress = atoms.get_stress()  # Stress tensor, shape (6,)
            num_atoms = len(atoms)

            # Accumulate total energy and forces
            self.total_energy += energy
            self.total_force += forces.sum(axis=0)  # Sum forces over all atoms
            self.total_stress += stress
            self.total_atoms += num_atoms
            self.total_structures += 1

        self.finalize()

    def finalize(self) -> None:
        """Compute the mean energy per atom, mean force vector, and mean stress tensor."""
        if self.total_atoms == 0 or self.total_structures == 0:
            raise ValueError("No data to finalize the model.")

        self.mean_energy_per_atom = self.total_energy / self.total_atoms
        self.mean_force = self.total_force / self.total_atoms
        self.mean_stress = self.total_stress / self.total_structures

    def predict_batch(
        self, batch_atoms: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict energies, forces, and stresses for a batch of structures.

        Args:
            batch_atoms (list): List of ASE Atoms objects.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Predicted energies, shape (batch_size,)
                - Predicted forces, shape (batch_size, max_num_atoms, 3)
                - Predicted stresses, shape (batch_size, 6)
        """
        predicted_energy_list: List[float] = []
        predicted_forces_list: List[np.ndarray] = []
        predicted_stress_list: List[np.ndarray] = []
        max_atoms = 0

        # Determine the maximum number of atoms in the batch for padding
        for atoms in batch_atoms:
            num_atoms = len(atoms)
            if num_atoms > max_atoms:
                max_atoms = num_atoms

        for atoms in batch_atoms:
            num_atoms = len(atoms)

            # Predict energy: mean energy per atom multiplied by number of atoms
            predicted_energy = self.mean_energy_per_atom * num_atoms
            predicted_energy_list.append(predicted_energy)

            # Predict forces: mean force vector for each atom
            predicted_forces = np.tile(
                self.mean_force, (num_atoms, 1)
            )  # Shape (num_atoms, 3)
            predicted_forces_list.append(predicted_forces)

            # Predict stress: same mean stress tensor for each structure
            predicted_stress = self.mean_stress.copy()
            predicted_stress_list.append(predicted_stress)

        # Pad forces to have the same number of atoms across the batch
        padded_forces = []
        for forces in predicted_forces_list:
            pad_width = (
                (0, max_atoms - forces.shape[0]),
                (0, 0),
            )  # Pad along the atom dimension
            padded = np.pad(forces, pad_width, mode="constant")
            padded_forces.append(padded)

        # Convert lists to PyTorch tensors
        predicted_energy_tensor = torch.tensor(
            predicted_energy_list, dtype=torch.float32
        )
        predicted_forces_tensor = torch.tensor(padded_forces, dtype=torch.float32)
        predicted_stress_tensor = torch.tensor(
            predicted_stress_list, dtype=torch.float32
        )

        return predicted_energy_tensor, predicted_forces_tensor, predicted_stress_tensor
