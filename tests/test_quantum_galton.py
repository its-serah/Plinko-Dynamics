"""
Test suite for QuantumGaltonBoard class.

Tests quantum circuit creation, simulation, and distribution generation.
"""

import pytest
import numpy as np
import pennylane as qml
from plinko_dynamics.core.quantum_galton import QuantumGaltonBoard


class TestQuantumGaltonBoard:
    """Test suite for QuantumGaltonBoard."""
    
    def test_initialization_valid(self):
        """Test valid initialization of QuantumGaltonBoard."""
        qgb = QuantumGaltonBoard(n_layers=3, n_shots=100)
        assert qgb.n_layers == 3
        assert qgb.n_shots == 100
        assert qgb.n_wires == 8  # (3 + 1) * 2
    
    def test_initialization_invalid_layers(self):
        """Test invalid n_layers raises ValueError."""
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            QuantumGaltonBoard(n_layers=0, n_shots=100)
    
    def test_initialization_invalid_shots(self):
        """Test invalid n_shots raises ValueError."""
        with pytest.raises(ValueError, match="n_shots must be > 0"):
            QuantumGaltonBoard(n_layers=3, n_shots=0)
    
    def test_gaussian_circuit_creation(self):
        """Test Gaussian circuit creation."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=100)
        circuit = qgb.create_gaussian_circuit(rotation_angle=0.3 * np.pi)
        assert callable(circuit)
        
        # Test circuit execution
        samples = circuit()
        assert samples.shape[0] == 100  # n_shots
        assert samples.shape[1] == 3  # n_layers + 1
    
    def test_exponential_circuit_creation(self):
        """Test exponential circuit creation."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=100)
        circuit = qgb.create_exponential_circuit(decay_rate=2.0)
        assert callable(circuit)
        
        # Test circuit execution
        samples = circuit()
        assert samples.shape[0] == 100
        assert samples.shape[1] == 3
    
    def test_exponential_circuit_invalid_decay(self):
        """Test exponential circuit with invalid decay rate."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=100)
        with pytest.raises(ValueError, match="Decay rate must be > 0"):
            qgb.create_exponential_circuit(decay_rate=-1.0)
    
    def test_hadamard_walk_circuit(self):
        """Test Hadamard quantum walk circuit."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=100)
        circuit = qgb.create_hadamard_walk_circuit()
        assert callable(circuit)
        
        samples = circuit()
        assert samples.shape[0] == 100
        assert samples.shape[1] == 3
    
    def test_run_simulation_gaussian(self):
        """Test running Gaussian simulation."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=1000)
        samples = qgb.run_simulation("gaussian", rotation_angle=0.3 * np.pi)
        
        assert len(samples) == 1000
        assert all(0 <= s <= 2 for s in samples)  # Check valid bin indices
    
    def test_run_simulation_invalid_type(self):
        """Test running simulation with invalid circuit type."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=100)
        with pytest.raises(ValueError, match="Unknown circuit type"):
            qgb.run_simulation("invalid_type")
    
    def test_get_probability_distribution(self):
        """Test probability distribution generation."""
        qgb = QuantumGaltonBoard(n_layers=3, n_shots=1000)
        samples = qgb.run_simulation("gaussian")
        distribution = qgb.get_probability_distribution(samples)
        
        assert len(distribution) == 4  # n_layers + 1
        assert np.isclose(np.sum(distribution), 1.0)  # Should sum to 1
        assert all(0 <= p <= 1 for p in distribution)  # Valid probabilities
    
    def test_reproducibility_with_seed(self):
        """Test reproducibility with fixed seed."""
        qgb1 = QuantumGaltonBoard(n_layers=2, n_shots=100, seed=42)
        samples1 = qgb1.run_simulation("gaussian")
        
        qgb2 = QuantumGaltonBoard(n_layers=2, n_shots=100, seed=42)
        samples2 = qgb2.run_simulation("gaussian")
        
        # Should produce identical results with same seed
        assert np.array_equal(samples1, samples2)
    
    def test_different_devices(self):
        """Test initialization with different quantum devices."""
        # Test with default device
        qgb_default = QuantumGaltonBoard(n_layers=2, n_shots=100, 
                                        device_name="default.qubit")
        assert qgb_default.device.name == "default.qubit"
        
        # Test with mixed device (if available)
        try:
            qgb_mixed = QuantumGaltonBoard(n_layers=2, n_shots=100,
                                          device_name="default.mixed")
            assert qgb_mixed.device.name == "default.mixed"
        except:
            pass  # Device might not be available
    
    def test_large_scale_simulation(self):
        """Test simulation with larger parameters."""
        qgb = QuantumGaltonBoard(n_layers=5, n_shots=5000)
        samples = qgb.run_simulation("gaussian")
        distribution = qgb.get_probability_distribution(samples)
        
        assert len(samples) == 5000
        assert len(distribution) == 6
        
        # Check for reasonable distribution shape (roughly bell-curved)
        max_idx = np.argmax(distribution)
        assert 1 <= max_idx <= 4  # Peak should be in middle bins
    
    @pytest.mark.parametrize("n_layers,expected_wires", [
        (1, 4),
        (2, 6),
        (3, 8),
        (4, 10),
        (5, 12)
    ])
    def test_wire_calculation(self, n_layers, expected_wires):
        """Test correct wire calculation for different layer counts."""
        qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=100)
        assert qgb.n_wires == expected_wires
    
    def test_circuit_metrics(self):
        """Test circuit complexity metrics."""
        qgb = QuantumGaltonBoard(n_layers=3, n_shots=100)
        metrics = qgb.get_circuit_metrics()
        
        assert "n_gates" in metrics
        assert "circuit_depth" in metrics
        assert "n_parameters" in metrics
        assert metrics["n_gates"] > 0
        assert metrics["circuit_depth"] > 0


class TestQuantumGaltonBoardIntegration:
    """Integration tests for QuantumGaltonBoard."""
    
    def test_multiple_circuit_types_comparison(self):
        """Test running multiple circuit types and comparing results."""
        qgb = QuantumGaltonBoard(n_layers=3, n_shots=1000)
        
        gaussian_samples = qgb.run_simulation("gaussian")
        exponential_samples = qgb.run_simulation("exponential")
        hadamard_samples = qgb.run_simulation("hadamard")
        
        # Get distributions
        gaussian_dist = qgb.get_probability_distribution(gaussian_samples)
        exponential_dist = qgb.get_probability_distribution(exponential_samples)
        hadamard_dist = qgb.get_probability_distribution(hadamard_samples)
        
        # All should be valid probability distributions
        for dist in [gaussian_dist, exponential_dist, hadamard_dist]:
            assert np.isclose(np.sum(dist), 1.0)
            assert all(0 <= p <= 1 for p in dist)
        
        # Distributions should be different
        assert not np.allclose(gaussian_dist, exponential_dist)
        assert not np.allclose(gaussian_dist, hadamard_dist)
    
    def test_noise_simulation(self):
        """Test simulation with noise model."""
        # Create a noisy device
        noise_prob = 0.01
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=1000)
        
        # Add noise to simulation
        samples_no_noise = qgb.run_simulation("gaussian")
        samples_with_noise = qgb.run_simulation("gaussian", noise_prob=noise_prob)
        
        # Distributions should be slightly different
        dist_no_noise = qgb.get_probability_distribution(samples_no_noise)
        dist_with_noise = qgb.get_probability_distribution(samples_with_noise)
        
        # Both should still be valid distributions
        assert np.isclose(np.sum(dist_no_noise), 1.0)
        assert np.isclose(np.sum(dist_with_noise), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
