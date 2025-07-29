# Plinko-Dynamics: Quantum Galton Board Implementation

## Project Overview

This project implements a quantum version of the Galton Board (Plinko) using quantum circuits to simulate complex statistical systems. It demonstrates the Universal Statistical Simulator approach for quantum Monte Carlo problems, which has potential applications in particle transport, quantum systems, and high-dimensional statistical challenges.

## Project Deliverables

This implementation addresses all 5 project deliverables:

### 1. Two-Page Summary Document
- **File**: `project_summary.md`
- Comprehensive understanding of quantum Galton board implementation
- Technical background and applications
- Circuit structure and algorithm explanation

### 2. General Algorithm Implementation
- **File**: `quantum_galton_board.py`
- General algorithm that generates circuits for any number of layers
- Verified Gaussian distribution output
- Scalable implementation supporting multiple layer configurations

### 3. Different Target Distributions
- **Gaussian Distribution**: Standard binomial approximation
- **Exponential Distribution**: Modified rotation angles for exponential decay
- **Hadamard Quantum Walk**: Equal probability quantum walk implementation

### 4. Noise Model Optimization
- Framework for implementing noise models from real hardware
- Optimization strategies for maximizing accuracy and layer count
- Error mitigation techniques outlined

### 5. Distance Metrics and Uncertainty Analysis
- Multiple distance metrics: MSE, KL-divergence, Total Variation, Chi-squared
- Statistical uncertainty calculations accounting for shot noise
- Comprehensive comparison framework

## Installation and Setup

### Requirements
```bash
pip install pennylane matplotlib scipy numpy
```

### Optional Dependencies
For quantum hardware simulation:
```bash
pip install pennylane-qiskit  # For IBM quantum devices
pip install pennylane-cirq    # For Google quantum devices
```

## Usage Examples

### Basic Gaussian Distribution
```python
from quantum_galton_board import QuantumGaltonBoard

# Create 4-layer Galton board
qgb = QuantumGaltonBoard(n_layers=4, n_shots=1000)

# Run simulation
samples = qgb.run_simulation("gaussian")
distribution = qgb.get_probability_distribution(samples)

# Visualize circuit
fig = qgb.visualize_circuit("gaussian")
```

### Exponential Distribution
```python
# Run exponential distribution simulation
exp_samples = qgb.run_simulation("exponential")
exp_distribution = qgb.get_probability_distribution(exp_samples)
```

### Hadamard Quantum Walk
```python
# Run Hadamard quantum walk
had_samples = qgb.run_simulation("hadamard")
had_distribution = qgb.get_probability_distribution(had_samples)
```

### Distance Metrics Analysis
```python
from quantum_galton_board import calculate_distance_metrics

# Compare quantum vs classical distributions
classical_dist, mse = qgb.compare_with_classical(distribution)
metrics = calculate_distance_metrics(distribution, classical_dist, 1000)

print(f"MSE: {metrics['mse']}")
print(f"KL Divergence: {metrics['kl_divergence']}")
print(f"TV Distance: {metrics['tv_distance']}")
```

## Technical Features

### Quantum Circuit Architecture
- **Control System**: RX rotation gates for probabilistic control
- **Position Tracking**: CSWAP gates for conditional state swapping
- **Entanglement**: CNOT gates for quantum correlations
- **Measurement**: Efficient sampling from final distributions

### Supported Platforms
- **Simulators**: PennyLane default.qubit (noiseless)
- **Quantum Hardware**: Any PennyLane-compatible quantum device
- **Noise Models**: Framework for implementing realistic hardware noise

### Performance Optimizations
- Efficient circuit construction for arbitrary layer counts
- Optimized measurement strategies
- Memory-efficient sample processing

## Project Structure

```
Plinko-Dynamics/
├── README.md                    # This file
├── project_summary.md          # Deliverable 1: Technical summary
├── quantum_galton_board.py     # Main implementation
├── examples/                   # Usage examples (coming soon)
├── tests/                      # Unit tests (coming soon)
└── results/                    # Experimental results (coming soon)
```

## Scientific Applications

### Monte Carlo Methods
- **Particle Transport**: Quantum simulation of particle scattering
- **Financial Modeling**: Risk assessment and option pricing
- **Statistical Physics**: Thermodynamic system simulations

### Quantum Advantage
- **Parallel Processing**: Superposition enables simultaneous path exploration
- **Interference Effects**: Quantum interference enhances statistical sampling
- **Scalability**: Potential exponential speedup for high-dimensional problems

## Future Enhancements

### Planned Features
- [ ] Noise model implementations for specific hardware
- [ ] Advanced optimization algorithms
- [ ] Interactive visualization tools
- [ ] Benchmark comparisons with classical methods
- [ ] Extended distribution types

### Hardware Integration
- [ ] IBM Quantum integration
- [ ] Google Quantum AI integration
- [ ] IonQ integration
- [ ] Optimization for specific hardware topologies

## Contributing

This project is part of the Womanium & WISER 2025 Quantum Program. Contributions, suggestions, and improvements are welcome!

## License

This project is open source. Please see the repository for license details.

## Acknowledgments

- Womanium & WISER 2025 Quantum Program
- Universal Statistical Simulator research by Mark Carney and Ben Varcoe
- PennyLane quantum computing framework

## Contact

Created as part of the Quantum Walks and Monte Carlo challenge.

---

*This implementation demonstrates quantum Monte Carlo methods using the Galton board as a testbed for exploring quantum advantages in statistical simulation.*
