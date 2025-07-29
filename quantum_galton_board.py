"""
Quantum Galton Board Implementation
A general algorithm for simulating a Galton Box using quantum circuits

This implementation is based on the Universal Statistical Simulator approach
for quantum Monte Carlo simulations relevant to particle transport and 
quantum systems.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np_vanilla
from typing import List, Tuple, Optional
import json

class QuantumGaltonBoard:
    """
    A quantum implementation of the Galton Board (Plinko) game.
    
    This class provides methods to create quantum circuits that simulate
    the statistical distribution of balls falling through a Galton board
    with pegs arranged in layers.
    """
    
    def __init__(self, n_layers: int, n_shots: int = 1000, device_name: str = "default.qubit"):
        """
        Initialize the Quantum Galton Board.
        
        Args:
            n_layers: Number of layers in the Galton board
            n_shots: Number of shots for sampling
            device_name: PennyLane device name
        """
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_probability_qubits = n_layers + 1
        self.half = self.n_probability_qubits
        self.n_wires = self.n_probability_qubits * 2
        
        # Measurement qubits (odd indices)
        self.measure_qubits = list(filter(lambda x: x % 2 != 0, list(range(self.n_wires))))
        
        # Create quantum device
        self.device = qml.device(device_name, wires=self.n_wires, shots=n_shots)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
    def create_circuit(self, rotation_angle: float = 0.26 * np.pi) -> qml.QNode:
        """
        Create the quantum circuit for the Galton board.
        
        Args:
            rotation_angle: Rotation angle for RX gates (controls distribution shape)
            
        Returns:
            QNode representing the quantum circuit
        """
        @qml.qnode(self.device)
        def circuit():
            # Initialize ancilla qubit
            qml.PauliX(wires=self.half)
            
            # Build layers of the Galton board
            for layer in range(1, self.n_layers + 1):
                # Reset and prepare control qubit for layers > 1
                if layer > 1:
                    qml.measure(wires=0)
                    qml.cond(0, qml.PauliX)(wires=0)
                
                # Apply rotation to control qubit
                qml.RX(rotation_angle, wires=0)
                
                # Calculate range for CSWAP operations in this layer
                start = self.half - layer
                end = self.half + layer - 1
                
                # Apply CSWAP and CNOT gates
                for j in range(start, end):
                    qml.CSWAP(wires=[0, j, j + 1])
                    qml.CNOT(wires=[j + 1, 0])
                
                # Final CSWAP operation for the layer
                qml.CSWAP(wires=[0, end + 1, end])
            
            return qml.sample(wires=self.measure_qubits)
        
        return circuit
    
    def create_exponential_circuit(self, decay_rate: float = 2.0) -> qml.QNode:
        """
        Create a quantum circuit that produces an exponential distribution.
        
        Args:
            decay_rate: Controls the decay rate of the exponential distribution
            
        Returns:
            QNode for exponential distribution
        """
        @qml.qnode(self.device)
        def circuit():
            # Initialize ancilla qubit
            qml.PauliX(wires=self.half)
            
            for layer in range(1, self.n_layers + 1):
                if layer > 1:
                    qml.measure(wires=0)
                    qml.cond(0, qml.PauliX)(wires=0)
                
                # Use different rotation angles to create exponential shape
                angle = np.arctan(np.exp(-decay_rate * layer / self.n_layers))
                qml.RX(angle, wires=0)
                
                start = self.half - layer
                end = self.half + layer - 1
                
                for j in range(start, end):
                    qml.CSWAP(wires=[0, j, j + 1])
                    qml.CNOT(wires=[j + 1, 0])
                
                qml.CSWAP(wires=[0, end + 1, end])
            
            return qml.sample(wires=self.measure_qubits)
        
        return circuit
    
    def create_hadamard_walk_circuit(self) -> qml.QNode:
        """
        Create a quantum circuit implementing Hadamard quantum walk.
        
        Returns:
            QNode for Hadamard quantum walk
        """
        @qml.qnode(self.device)
        def circuit():
            # Initialize with Hadamard gates for symmetric walk
            qml.PauliX(wires=self.half)
            
            for layer in range(1, self.n_layers + 1):
                if layer > 1:
                    qml.measure(wires=0)
                    qml.cond(0, qml.PauliX)(wires=0)
                
                # Use Hadamard gate for equal probability
                qml.Hadamard(wires=0)
                
                start = self.half - layer
                end = self.half + layer - 1
                
                for j in range(start, end):
                    qml.CSWAP(wires=[0, j, j + 1])
                    qml.CNOT(wires=[j + 1, 0])
                
                qml.CSWAP(wires=[0, end + 1, end])
            
            return qml.sample(wires=self.measure_qubits)
        
        return circuit
    
    def run_simulation(self, circuit_type: str = "gaussian") -> np.ndarray:
        """
        Run the quantum simulation.
        
        Args:
            circuit_type: Type of circuit ("gaussian", "exponential", "hadamard")
            
        Returns:
            Array of measurement results
        """
        if circuit_type == "gaussian":
            circuit = self.create_circuit()
        elif circuit_type == "exponential":
            circuit = self.create_exponential_circuit()
        elif circuit_type == "hadamard":
            circuit = self.create_hadamard_walk_circuit()
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Execute circuit
        samples = circuit()
        return samples
    
    def get_probability_distribution(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert measurement samples to probability distribution.
        
        Args:
            samples: Raw measurement samples
            
        Returns:
            Probability distribution array
        """
        total = np.sum(samples, axis=0)
        return total / self.n_shots
    
    def visualize_circuit(self, circuit_type: str = "gaussian"):
        """
        Visualize the quantum circuit.
        
        Args:
            circuit_type: Type of circuit to visualize
        """
        if circuit_type == "gaussian":
            circuit = self.create_circuit()
        elif circuit_type == "exponential":
            circuit = self.create_exponential_circuit()
        elif circuit_type == "hadamard":
            circuit = self.create_hadamard_walk_circuit()
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        fig, ax = qml.draw_mpl(circuit)()
        fig.suptitle(f"Quantum Galton Board - {circuit_type.title()} Distribution")
        plt.tight_layout()
        return fig
    
    def compare_with_classical(self, quantum_dist: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compare quantum results with classical Galton board simulation.
        
        Args:
            quantum_dist: Quantum probability distribution
            
        Returns:
            Tuple of (classical_distribution, mse_error)
        """
        classical_dist = self._classical_galton_simulation()
        mse_error = np.mean((quantum_dist - classical_dist) ** 2)
        return classical_dist, mse_error
    
    def _classical_galton_simulation(self) -> np.ndarray:
        """
        Classical Galton board simulation for comparison.
        
        Returns:
            Classical probability distribution
        """
        containers = [0] * (self.n_layers + 1)
        
        for _ in range(self.n_shots):
            pos = 0
            for _ in range(self.n_layers):
                if np.random.random() < 0.5:
                    pos += 1
            containers[pos] += 1
        
        return np.array(containers) / self.n_shots


def classical_galton_board(n_layers: int, n_shots: int) -> List[float]:
    """
    Classical implementation of Galton board for comparison.
    
    Args:
        n_layers: Number of layers
        n_shots: Number of balls
        
    Returns:
        Probability distribution
    """
    from random import randint
    
    containers = [0] * (n_layers + 1)
    for _ in range(n_shots):
        pos = 0
        for _ in range(n_layers):
            turn = randint(0, 1)
            if turn == 1:
                pos += 1
        containers[pos] += 1
    
    return [val / n_shots for val in containers]


def calculate_distance_metrics(quantum_dist: np.ndarray, 
                             target_dist: np.ndarray, 
                             n_shots: int) -> dict:
    """
    Calculate various distance metrics between distributions.
    
    Args:
        quantum_dist: Quantum distribution
        target_dist: Target distribution
        n_shots: Number of shots (for uncertainty calculation)
        
    Returns:
        Dictionary of distance metrics
    """
    # Mean Squared Error
    mse = np.mean((quantum_dist - target_dist) ** 2)
    
    # Kullback-Leibler Divergence
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = np.sum(target_dist * np.log((target_dist + eps) / (quantum_dist + eps)))
    
    # Total Variation Distance
    tv_distance = 0.5 * np.sum(np.abs(quantum_dist - target_dist))
    
    # Chi-squared distance
    chi_squared = np.sum((quantum_dist - target_dist) ** 2 / (target_dist + eps))
    
    # Statistical uncertainty (assuming Poisson statistics)
    quantum_uncertainty = np.sqrt(quantum_dist * (1 - quantum_dist) / n_shots)
    target_uncertainty = np.sqrt(target_dist * (1 - target_dist) / n_shots)
    
    return {
        'mse': mse,
        'kl_divergence': kl_div,
        'tv_distance': tv_distance,
        'chi_squared': chi_squared,
        'quantum_uncertainty': quantum_uncertainty.tolist(),
        'target_uncertainty': target_uncertainty.tolist()
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Quantum Galton Board Implementation")
    print("=" * 50)
    
    # Test with different layer counts
    for n_layers in [2, 3, 4]:
        print(f"\nTesting {n_layers} layers:")
        
        # Create quantum Galton board
        qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=1000)
        
        # Run Gaussian simulation
        samples = qgb.run_simulation("gaussian")
        quantum_dist = qgb.get_probability_distribution(samples)
        
        # Compare with classical
        classical_dist, mse = qgb.compare_with_classical(quantum_dist)
        
        print(f"Quantum distribution:  {quantum_dist}")
        print(f"Classical distribution: {classical_dist}")
        print(f"MSE: {mse:.6f}")
        
        # Calculate distance metrics
        metrics = calculate_distance_metrics(quantum_dist, classical_dist, 1000)
        print(f"KL Divergence: {metrics['kl_divergence']:.6f}")
        print(f"TV Distance: {metrics['tv_distance']:.6f}")
