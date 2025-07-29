"""
AI Data Generator for Quantum Galton Board
Generates training data from quantum circuit simulations for AI model training
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from quantum_galton_board import QuantumGaltonBoard
import json
from typing import List, Tuple, Optional

class GaltonTrajectoryDataset(Dataset):
    """
    Dataset for quantum Galton board trajectories.
    
    This class generates training data by running multiple quantum simulations
    with different parameters and collecting the resulting probability distributions
    over time steps.
    """
    
    def __init__(self, 
                 num_trajectories: int = 100,
                 max_layers: int = 6,
                 min_layers: int = 2,
                 n_shots: int = 1000,
                 time_evolution_steps: int = 50,
                 circuit_types: List[str] = ["gaussian", "exponential", "hadamard"],
                 save_path: Optional[str] = None):
        """
        Initialize the dataset generator.
        
        Args:
            num_trajectories: Number of trajectory samples to generate
            max_layers: Maximum number of Galton board layers
            min_layers: Minimum number of Galton board layers
            n_shots: Number of quantum measurements per simulation
            time_evolution_steps: Number of time steps in each trajectory
            circuit_types: Types of quantum circuits to include
            save_path: Path to save generated dataset
        """
        self.num_trajectories = num_trajectories
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.n_shots = n_shots
        self.time_evolution_steps = time_evolution_steps
        self.circuit_types = circuit_types
        
        # Generate the dataset
        self.trajectories, self.parameters, self.metadata = self._generate_trajectories()
        
        # Convert to tensors
        self.trajectories = torch.tensor(self.trajectories, dtype=torch.float32)
        self.parameters = torch.tensor(self.parameters, dtype=torch.float32)
        
        # Save if path provided
        if save_path:
            self.save_dataset(save_path)
    
    def _generate_trajectories(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate quantum Galton board trajectories with varying parameters."""
        trajectories = []
        parameters = []
        metadata = {
            'circuit_types': [],
            'layer_counts': [],
            'rotation_angles': [],
            'decay_rates': []
        }
        
        print(f"Generating {self.num_trajectories} quantum trajectories...")
        
        for i in range(self.num_trajectories):
            # Randomly sample parameters
            n_layers = np.random.randint(self.min_layers, self.max_layers + 1)
            circuit_type = np.random.choice(self.circuit_types)
            rotation_angle = np.random.uniform(0.1 * np.pi, 0.5 * np.pi)
            decay_rate = np.random.uniform(0.5, 3.0)
            
            # Create quantum Galton board
            qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=self.n_shots)
            
            # Generate trajectory by running circuit multiple times with noise
            trajectory = self._generate_single_trajectory(qgb, circuit_type, 
                                                        rotation_angle, decay_rate)
            
            trajectories.append(trajectory)
            parameters.append([n_layers, rotation_angle, decay_rate])
            
            # Store metadata
            metadata['circuit_types'].append(circuit_type)
            metadata['layer_counts'].append(n_layers)
            metadata['rotation_angles'].append(rotation_angle)
            metadata['decay_rates'].append(decay_rate)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{self.num_trajectories} trajectories")
        
        return np.array(trajectories), np.array(parameters), metadata
    
    def _generate_single_trajectory(self, qgb, circuit_type, rotation_angle, decay_rate):
        """Generate a single trajectory by evolving quantum state over time."""
        trajectory = []
        
        # Simulate time evolution by adding noise and running circuit
        for t in range(self.time_evolution_steps):
            # Add time-dependent rotation angle
            current_angle = rotation_angle * (1 + 0.1 * np.sin(t * 0.1))
            
            # Modify the circuit based on type and current parameters
            if circuit_type == "gaussian":
                circuit = qgb.create_circuit(current_angle)
            elif circuit_type == "exponential":
                current_decay = decay_rate * (1 + 0.05 * t)
                circuit = qgb.create_exponential_circuit(current_decay)
            else:  # hadamard
                circuit = qgb.create_hadamard_walk_circuit()
            
            # Run simulation
            samples = circuit()
            distribution = qgb.get_probability_distribution(samples)
            
            # Pad or truncate to fixed size for consistent tensor shapes
            max_bins = self.max_layers + 1
            if len(distribution) < max_bins:
                padded_dist = np.zeros(max_bins)
                padded_dist[:len(distribution)] = distribution
                distribution = padded_dist
            elif len(distribution) > max_bins:
                distribution = distribution[:max_bins]
            
            trajectory.append(distribution)
        
        return np.array(trajectory)
    
    def save_dataset(self, path: str):
        """Save the generated dataset."""
        dataset_dict = {
            'trajectories': self.trajectories.numpy(),
            'parameters': self.parameters.numpy(),
            'metadata': self.metadata,
            'config': {
                'num_trajectories': self.num_trajectories,
                'max_layers': self.max_layers,
                'min_layers': self.min_layers,
                'n_shots': self.n_shots,
                'time_evolution_steps': self.time_evolution_steps,
                'circuit_types': self.circuit_types
            }
        }
        
        torch.save(dataset_dict, path)
        print(f"Dataset saved to {path}")
    
    @classmethod
    def load_dataset(cls, path: str):
        """Load a previously saved dataset."""
        data = torch.load(path)
        
        # Create empty instance
        instance = cls.__new__(cls)
        
        # Load data
        instance.trajectories = torch.tensor(data['trajectories'], dtype=torch.float32)
        instance.parameters = torch.tensor(data['parameters'], dtype=torch.float32)
        instance.metadata = data['metadata']
        
        # Load config
        config = data['config']
        instance.num_trajectories = config['num_trajectories']
        instance.max_layers = config['max_layers']
        instance.min_layers = config['min_layers']
        instance.n_shots = config['n_shots']
        instance.time_evolution_steps = config['time_evolution_steps']
        instance.circuit_types = config['circuit_types']
        
        return instance
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.parameters[idx]
    
    def get_trajectory_info(self, idx: int) -> dict:
        """Get detailed information about a specific trajectory."""
        return {
            'circuit_type': self.metadata['circuit_types'][idx],
            'n_layers': self.metadata['layer_counts'][idx],
            'rotation_angle': self.metadata['rotation_angles'][idx],
            'decay_rate': self.metadata['decay_rates'][idx],
            'trajectory_shape': self.trajectories[idx].shape,
            'parameters': self.parameters[idx].numpy()
        }


def generate_training_data(num_samples: int = 50, save_path: str = "galton_ai_dataset.pt"):
    """
    Generate training data for AI models.
    
    Args:
        num_samples: Number of trajectory samples
        save_path: Path to save the dataset
    
    Returns:
        GaltonTrajectoryDataset instance
    """
    print("Generating quantum Galton board training data for AI models...")
    
    dataset = GaltonTrajectoryDataset(
        num_trajectories=num_samples,
        max_layers=5,
        min_layers=2,
        n_shots=500,  # Reduced for faster generation
        time_evolution_steps=20,
        save_path=save_path
    )
    
    print(f"Generated dataset with shape: {dataset.trajectories.shape}")
    print(f"Parameter tensor shape: {dataset.parameters.shape}")
    
    return dataset


if __name__ == "__main__":
    # Generate a small test dataset
    dataset = generate_training_data(num_samples=20, save_path="test_galton_dataset.pt")
    
    # Print some info
    print("\nDataset info:")
    print(f"Total trajectories: {len(dataset)}")
    print(f"Trajectory shape: {dataset.trajectories[0].shape}")
    print(f"First trajectory info: {dataset.get_trajectory_info(0)}")
