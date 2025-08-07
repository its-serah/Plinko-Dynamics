"""
Enhanced Quantum Galton Board Implementation

A modular, robust implementation of quantum Galton board simulation
with comprehensive error handling and validation.
"""

import pennylane as qml
from pennylane import numpy as np
import numpy as np_vanilla
from typing import List, Tuple, Optional, Dict, Any
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumGaltonBoard:
    """
    A quantum implementation of the Galton Board (Plinko) game.
    
    This class provides methods to create quantum circuits that simulate
    the statistical distribution of balls falling through a Galton board
    with pegs arranged in layers.
    
    Attributes:
        n_layers (int): Number of layers in the Galton board
        n_shots (int): Number of quantum measurements
        device (qml.Device): PennyLane quantum device
        n_wires (int): Total number of qubits needed
    """
    
    def __init__(self, 
                 n_layers: int, 
                 n_shots: int = 1000, 
                 device_name: str = "default.qubit",
                 seed: Optional[int] = None):
        """
        Initialize the Quantum Galton Board.
        
        Args:
            n_layers: Number of layers in the Galton board (must be >= 1)
            n_shots: Number of shots for sampling (must be > 0)
            device_name: PennyLane device name
            seed: Random seed for reproducibility
            
        Raises:
            ValueError: If n_layers < 1 or n_shots <= 0
            RuntimeError: If device initialization fails
        """
        # Validate inputs
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_shots <= 0:
            raise ValueError(f"n_shots must be > 0, got {n_shots}")
            
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_probability_qubits = n_layers + 1
        self.half = self.n_probability_qubits
        self.n_wires = self.n_probability_qubits * 2
        
        # Measurement qubits (odd indices)
        self.measure_qubits = list(range(1, self.n_wires, 2))
        
        try:
            # Create quantum device
            self.device = qml.device(device_name, wires=self.n_wires, shots=n_shots)
            logger.info(f"Initialized quantum device '{device_name}' with {self.n_wires} wires")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize quantum device: {e}")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            np_vanilla.random.seed(seed)
            
    def _validate_rotation_angle(self, angle: float) -> None:
        """Validate rotation angle is within reasonable bounds."""
        if not (0 <= angle <= 2 * np.pi):
            warnings.warn(f"Rotation angle {angle} outside [0, 2π] range")
            
    def create_gaussian_circuit(self, rotation_angle: float = 0.26 * np.pi) -> qml.QNode:
        """
        Create the quantum circuit for Gaussian-like distribution.
        
        Args:
            rotation_angle: Rotation angle for RX gates (controls distribution shape)
            
        Returns:
            QNode representing the quantum circuit
            
        Raises:
            ValueError: If rotation angle is invalid
        """
        self._validate_rotation_angle(rotation_angle)
        
        @qml.qnode(self.device)
        def circuit():
            try:
                # Initialize ancilla qubit
                qml.PauliX(wires=self.half)
                
                # Build layers of the Galton board
                for layer in range(1, self.n_layers + 1):
                    # Reset and prepare control qubit for layers > 1
                    if layer > 1:
                        # Use proper conditional measurement
                        m = qml.measure(wires=0)
                        qml.cond(m, qml.PauliX)(wires=0)
                    
                    # Apply rotation to control qubit
                    qml.RX(rotation_angle, wires=0)
                    
                    # Calculate range for CSWAP operations in this layer
                    start = max(1, self.half - layer)  # Ensure we don't go below wire 1
                    end = min(self.n_wires - 2, self.half + layer - 1)  # Ensure we don't exceed wire limits
                    
                    # Apply CSWAP and CNOT gates
                    for j in range(start, end):
                        if j + 1 < self.n_wires:  # Safety check
                            qml.CSWAP(wires=[0, j, j + 1])
                            qml.CNOT(wires=[j + 1, 0])
                    
                    # Final CSWAP operation for the layer
                    if end + 1 < self.n_wires and end >= 1:
                        qml.CSWAP(wires=[0, end + 1, end])
                
                return qml.sample(wires=self.measure_qubits)
            except Exception as e:
                logger.error(f"Error in quantum circuit execution: {e}")
                raise
        
        return circuit
    
    def create_exponential_circuit(self, decay_rate: float = 2.0) -> qml.QNode:
        """
        Create a quantum circuit that produces an exponential-like distribution.
        
        Args:
            decay_rate: Controls the decay rate of the exponential distribution
            
        Returns:
            QNode for exponential distribution
        """
        if decay_rate <= 0:
            raise ValueError(f"Decay rate must be > 0, got {decay_rate}")
            
        @qml.qnode(self.device)
        def circuit():
            try:
                # Initialize ancilla qubit
                qml.PauliX(wires=self.half)
                
                for layer in range(1, self.n_layers + 1):
                    if layer > 1:
                        m = qml.measure(wires=0)
                        qml.cond(m, qml.PauliX)(wires=0)
                    
                    # Use different rotation angles to create exponential shape
                    angle = np.arctan(np.exp(-decay_rate * layer / self.n_layers))
                    qml.RX(angle, wires=0)
                    
                    start = max(1, self.half - layer)
                    end = min(self.n_wires - 2, self.half + layer - 1)
                    
                    for j in range(start, end):
                        if j + 1 < self.n_wires:
                            qml.CSWAP(wires=[0, j, j + 1])
                            qml.CNOT(wires=[j + 1, 0])
                    
                    if end + 1 < self.n_wires and end >= 1:
                        qml.CSWAP(wires=[0, end + 1, end])
                
                return qml.sample(wires=self.measure_qubits)
            except Exception as e:
                logger.error(f"Error in exponential circuit execution: {e}")
                raise
        
        return circuit
    
    def create_hadamard_walk_circuit(self) -> qml.QNode:
        """
        Create a quantum circuit implementing Hadamard quantum walk.
        
        Returns:
            QNode for Hadamard quantum walk
        """
        @qml.qnode(self.device)
        def circuit():
            try:
                # Initialize with Hadamard gates for symmetric walk
                qml.PauliX(wires=self.half)
                
                for layer in range(1, self.n_layers + 1):
                    if layer > 1:
                        m = qml.measure(wires=0)
                        qml.cond(m, qml.PauliX)(wires=0)
                    
                    # Use Hadamard gate for equal probability
                    qml.Hadamard(wires=0)
                    
                    start = max(1, self.half - layer)
                    end = min(self.n_wires - 2, self.half + layer - 1)
                    
                    for j in range(start, end):
                        if j + 1 < self.n_wires:
                            qml.CSWAP(wires=[0, j, j + 1])
                            qml.CNOT(wires=[j + 1, 0])
                    
                    if end + 1 < self.n_wires and end >= 1:
                        qml.CSWAP(wires=[0, end + 1, end])
                
                return qml.sample(wires=self.measure_qubits)
            except Exception as e:
                logger.error(f"Error in Hadamard walk circuit execution: {e}")
                raise
        
        return circuit
    
    def run_simulation(self, circuit_type: str = "gaussian", **kwargs) -> np.ndarray:
        """
        Run the quantum simulation.
        
        Args:
            circuit_type: Type of circuit ("gaussian", "exponential", "hadamard")
            **kwargs: Additional parameters for specific circuit types
            
        Returns:
            Array of measurement results
            
        Raises:
            ValueError: If unknown circuit type is specified
            RuntimeError: If simulation execution fails
        """
        circuit_map = {
            "gaussian": self.create_gaussian_circuit,
            "exponential": self.create_exponential_circuit,
            "hadamard": self.create_hadamard_walk_circuit
        }
        
        if circuit_type not in circuit_map:
            raise ValueError(f"Unknown circuit type: {circuit_type}. "
                           f"Available types: {list(circuit_map.keys())}")
        
        try:
            # Create circuit with appropriate parameters
            if circuit_type == "gaussian":
                circuit = circuit_map[circuit_type](kwargs.get('rotation_angle', 0.26 * np.pi))
            elif circuit_type == "exponential":
                circuit = circuit_map[circuit_type](kwargs.get('decay_rate', 2.0))
            else:
                circuit = circuit_map[circuit_type]()
            
            # Execute circuit
            samples = circuit()
            logger.info(f"Successfully executed {circuit_type} simulation with {len(samples)} samples")
            return samples
            
        except Exception as e:
            logger.error(f"Simulation failed for {circuit_type}: {e}")
            raise RuntimeError(f"Simulation execution failed: {e}")
    
    def get_probability_distribution(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert measurement samples to probability distribution.
        
        Args:
            samples: Raw measurement samples
            
        Returns:
            Probability distribution array
            
        Raises:
            ValueError: If samples array is invalid
        """
        if samples is None or len(samples) == 0:
            raise ValueError("Samples array cannot be empty")
            
        try:
            # Sum across shots to get bin counts
            if samples.ndim == 1:
                # Single measurement case
                total = samples
            else:
                # Multiple measurements case
                total = np.sum(samples, axis=0)
            
            # Normalize to get probabilities
            prob_dist = total / self.n_shots
            
            # Ensure probabilities sum to approximately 1
            total_prob = np.sum(prob_dist)
            if not np.isclose(total_prob, 1.0, atol=0.1):
                warnings.warn(f"Probability distribution sums to {total_prob}, expected ~1.0")
            
            return prob_dist
        except Exception as e:
            raise ValueError(f"Failed to compute probability distribution: {e}")
    
    def get_circuit_info(self, circuit_type: str = "gaussian") -> Dict[str, Any]:
        """
        Get information about the quantum circuit.
        
        Args:
            circuit_type: Type of circuit to analyze
            
        Returns:
            Dictionary containing circuit information
        """
        circuit_info = {
            "n_layers": self.n_layers,
            "n_shots": self.n_shots,
            "n_wires": self.n_wires,
            "n_probability_qubits": self.n_probability_qubits,
            "measure_qubits": self.measure_qubits,
            "circuit_type": circuit_type,
            "expected_distribution_size": self.n_layers + 1
        }
        
        try:
            # Get circuit depth if possible
            if circuit_type == "gaussian":
                circuit = self.create_gaussian_circuit()
            elif circuit_type == "exponential":
                circuit = self.create_exponential_circuit()
            elif circuit_type == "hadamard":
                circuit = self.create_hadamard_walk_circuit()
            else:
                return circuit_info
                
            # Try to get circuit specs (this might not work with all PennyLane versions)
            try:
                specs = qml.specs(circuit)()
                circuit_info.update({
                    "depth": specs.get("depth", "unknown"),
                    "num_operations": specs.get("num_operations", "unknown"),
                    "gate_types": specs.get("gate_types", {})
                })
            except:
                pass  # Specs not available
                
        except Exception as e:
            logger.warning(f"Could not analyze circuit: {e}")
        
        return circuit_info
