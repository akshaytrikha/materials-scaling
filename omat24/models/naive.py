import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

class NaiveAtomModel:
    def __init__(self, max_neighbors=0):
        self.max_neighbors = max_neighbors  # Number of neighbors to consider
        self.force_sum = {}
        self.force_count = {}
        self.energy_sum = {}
        self.energy_count = {}
        self.stress_sum = np.zeros(6)
        self.stress_count = 0
    
    @staticmethod
    def zero_vector():
        return np.zeros(3)

    def train(self, dataset):
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

            # Find nearest neighbors (up to max_neighbors)
            nearest_indices = self.find_nearest_neighbors(distance_matrix)

            # Estimate per-atom energy contribution
            energy_per_atom = energy / num_atoms if num_atoms > 0 else 0.0
            for idx, atom in enumerate(atoms):
                atom_type = symbols[idx]
                    
                # Get the index of the nearest neighbor for the current atom
                neighbor_idx = nearest_indices[idx]
                    
                # Check if the neighbor_idx is within the valid range of indices
                if isinstance(neighbor_idx, int) and 0 <= neighbor_idx < len(symbols):
                    neighbor_type = symbols[neighbor_idx]  # Get the neighbor's chemical symbol
                else:
                    neighbor_type = None  # Handle cases where no valid neighbor exists

                # Initialize dictionaries for the current atom type
                if atom_type not in self.force_sum:
                    self.force_sum[atom_type] = defaultdict(self.zero_vector)
                if atom_type not in self.force_count:
                    self.force_count[atom_type] = defaultdict(int)
                
                # Initialize energy accumulation dictionaries
                if atom_type not in self.energy_sum:
                    self.energy_sum[atom_type] = defaultdict(float)
                if atom_type not in self.energy_count:
                    self.energy_count[atom_type] = defaultdict(int)

                # If neighbor exists, accumulate values; otherwise, back off to atom_type only
                key = neighbor_type if neighbor_type is not None else "atom_type"
                
                # Accumulate forces
                self.force_sum[atom_type][key] += forces[idx]
                self.force_count[atom_type][key] += 1
                
                # Accumulate energies
                self.energy_sum[atom_type][key] += energy_per_atom
                self.energy_count[atom_type][key] += 1

            # Accumulate stress
            self.stress_sum += stress
            self.stress_count += 1


    def find_nearest_neighbors(self, distance_matrix):
        """Find indices of up to max_neighbors nearest neighbors for each atom."""
        nearest_indices = []
        for i, distances in enumerate(distance_matrix):
            # Exclude self-distance by setting it to infinity
            distances = distances.copy()
            distances[i] = np.inf
            sorted_indices = np.argsort(distances)[:self.max_neighbors]
            nearest_indices.append(sorted_indices)
        return nearest_indices

    def finalize(self):
        """Compute mean forces and energies."""
        self.mean_forces = {}
        self.mean_energies = {}

        for atom_type in self.force_sum.keys():
            self.mean_forces[atom_type] = {}  # Initialize the dictionary for each atom_type
            
            for neighbor_type in self.force_sum[atom_type].keys():
                # Ensure the key exists before accessing
                if neighbor_type not in self.mean_forces[atom_type]:
                    self.mean_forces[atom_type][neighbor_type] = 0  # Initialize with default value
                
                # Now you can safely compute and store the average
                force_sum = self.force_sum[atom_type][neighbor_type]
                count = self.force_count[atom_type][neighbor_type]
                self.mean_forces[atom_type][neighbor_type] = force_sum / count

            # You can add additional logic to handle default values for atom_type
            if atom_type not in self.mean_forces:
                self.mean_forces[atom_type]["atom_type"] = 0  # Initialize with default

        for atom_type in self.energy_sum.keys():
            self.mean_energies[atom_type] = {}  # Initialize the dictionary for each atom_type
            
            for neighbor_type in self.energy_sum[atom_type].keys():
                # Ensure the key exists before accessing
                if neighbor_type not in self.mean_energies[atom_type]:
                    self.mean_energies[atom_type][neighbor_type] = 0  # Initialize with default value
                
                # Now you can safely compute and store the average
                energy_sum = self.energy_sum[atom_type][neighbor_type]
                count = self.energy_count[atom_type][neighbor_type]
                self.mean_energies[atom_type][neighbor_type] = energy_sum / count

            # You can add additional logic to handle default values for atom_type
            if atom_type not in self.mean_energies:
                self.mean_energies[atom_type]["atom_type"] = 0  # Initialize with default value


        # Mean stress
        self.mean_stress = (
            self.stress_sum / self.stress_count if self.stress_count > 0 else np.zeros(6)
        )

    def predict(self, atoms):
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        num_atoms = len(atoms)

        if num_atoms == 0:
            return 0.0, np.zeros((0, 3)), self.mean_stress

        # Compute distance matrix
        distance_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )

        # Find nearest neighbors (up to max_neighbors)
        nearest_indices = self.find_nearest_neighbors(distance_matrix)

        predicted_forces = np.zeros((num_atoms, 3))
        predicted_energy = 0.0

        for idx, atom in enumerate(atoms):
            atom_type = symbols[idx]
            neighbors = [symbols[n_idx] for n_idx in nearest_indices[idx]]

            # Predict using up to max_neighbors, back off as needed
            predicted_force = None
            predicted_atom_energy = 0.0

            # Try to find a prediction for each possible number of neighbors, starting from max_neighbors
            for num_neighbors in range(self.max_neighbors, 0, -1):
                key = tuple(neighbors[:num_neighbors])  # Consider neighbors up to num_neighbors
                if key in self.mean_forces[atom_type]:
                    predicted_force = self.mean_forces[atom_type][key]
                    predicted_atom_energy = self.mean_energies[atom_type].get(key, 0.0)
                    break

            # If no prediction was found (i.e., key not found), use the force for a single atom
            if predicted_force is None:
                predicted_force = self.mean_forces[atom_type].get("atom_type", np.zeros(3))
                predicted_atom_energy = self.mean_energies[atom_type].get("atom_type", 0.0)

            # Accumulate the force and energy predictions
            predicted_forces[idx] = predicted_force
            predicted_energy += predicted_atom_energy

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