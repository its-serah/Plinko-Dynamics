# Plinko Dynamics v2.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.32%2B-orange.svg)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()


# Team Members
- **Sarah Rashidi**:  gst-Hp9AXRvC2Scd65A
- **Elena Bresta**:   gst-uAhuv0ZAs1mBqiq
- **Krishang Gupta**: gst-QwPcAHbHcHFGkF2 


## Advanced Quantum Galton Board Simulation with AI-Enhanced Analysis

A sophisticated, modular Python package for simulating quantum Galton boards (Plinko) with beautiful visualizations, comprehensive error handling, and AI-powered trajectory analysis.

## Features

### Quantum Simulation
- **Multiple Circuit Types**: Gaussian, exponential, and Hadamard quantum walks
- **Robust Error Handling**: Comprehensive validation and logging
- **Hardware Ready**: Compatible with quantum hardware backends
- **Scalable Architecture**: Efficient for large-scale simulations

### Beautiful Visualizations
- **Purple & Navy Theme**: Stunning color schemes for professional presentations
- **Interactive Plots**: Comprehensive analysis dashboards
- **Publication Ready**: High-DPI exports for papers and reports
- **Real-time Updates**: Dynamic visualization capabilities

### AI-Enhanced Analysis
- **Neural ODEs**: Advanced trajectory modeling with torchdiffeq
- **Latent Space Learning**: Compact representation of quantum dynamics
- **Predictive Models**: Forecast distribution evolution
- **Robust Training**: Gradient clipping and advanced optimization

### Comprehensive Metrics
- **Statistical Analysis**: 12+ distance measures and overlap metrics
- **Performance Comparison**: Circuit ranking and optimization guidance
- **Time Series Analysis**: Trajectory stability and convergence metrics
- **Quantum Advantage**: Specialized quantum vs classical comparisons

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/its-serah/Plinko-Dynamics.git
cd Plinko-Dynamics

# Install dependencies
pip install -r requirements.txt

# Install in development mode (recommended)
pip install -e .
```

### Basic Usage

```python
from plinko_dynamics import QuantumGaltonBoard, PlinkoDynamicsVisualizer

# Create quantum Galton board
qgb = QuantumGaltonBoard(n_layers=4, n_shots=1000)

# Run different circuit types
gaussian_samples = qgb.run_simulation("gaussian", rotation_angle=0.3 * np.pi)
exponential_samples = qgb.run_simulation("exponential", decay_rate=2.0)
hadamard_samples = qgb.run_simulation("hadamard")

# Get probability distributions
gaussian_dist = qgb.get_probability_distribution(gaussian_samples)

# Create beautiful visualizations
visualizer = PlinkoDynamicsVisualizer(theme='purple_navy')
fig = visualizer.create_beautiful_plinko_board(
    n_layers=4, 
    distribution=gaussian_dist,
    title="Quantum Plinko Dynamics"
)
```

### Enhanced Demo

```bash
# Run the comprehensive demo
python3 demo_enhanced.py
```

This will generate:
- Quantum circuit performance analysis
- Quantum vs classical comparisons  
- Beautiful Plinko board visualizations
- AI trajectory evolution analysis
- Comprehensive metrics reports

## Architecture

### Modular Design

```
plinko_dynamics/
├── core/                   # Core simulation engines
│   ├── quantum_galton.py   # Quantum circuit implementation
│   └── classical_simulator.py  # Classical comparison
├── ai/                     # AI models and training
│   └── models.py          # Neural ODE and RNN models
├── visualization/          # Beautiful plotting
│   ├── plotter.py         # Main visualization class
│   └── themes.py          # Color themes and styles
└── utils/                  # Utilities and metrics
    ├── metrics.py         # Statistical analysis
    └── data_generation.py # Dataset generation
```

### Key Components

#### QuantumGaltonBoard
- Robust quantum circuit implementation
- Multiple distribution types
- Comprehensive error handling
- Hardware backend compatibility

#### PlinkoDynamicsVisualizer
- Purple & navy color themes
- Publication-ready plots
- Interactive dashboards
- Real-time visualization

#### QuantumGaltonAI
- Neural ODE trajectory modeling
- Latent space representation
- Predictive capabilities
- Robust training pipeline

#### DistributionMetrics
- 12+ distance measures
- Statistical hypothesis testing
- Quantum advantage analysis
- Time series evaluation

## Advanced Usage

### AI Trajectory Analysis

```python
from plinko_dynamics import QuantumGaltonAI

# Initialize AI model
ai_model = QuantumGaltonAI(
    obs_dim=5,          # Number of bins
    latent_dim=4,       # Latent space dimension
    nhidden=32,         # Hidden units
    lr=1e-3            # Learning rate
)

# Train on trajectory data
training_history = ai_model.train(
    trajectories=trajectory_data,
    time_steps=time_array,
    num_epochs=50
)
```

### Comprehensive Metrics Analysis

```python
from plinko_dynamics.utils import DistributionMetrics

# Quantum vs classical analysis
analysis = DistributionMetrics.quantum_classical_analysis(
    quantum_dist=quantum_distribution,
    classical_dist=classical_distribution,
    theoretical_dist=theoretical_binomial
)

# Circuit performance comparison
performance = DistributionMetrics.circuit_performance_metrics(
    distributions={
        "gaussian": gaussian_dist,
        "exponential": exponential_dist,
        "hadamard": hadamard_dist
    },
    target_dist=target_distribution
)
```

### Custom Visualizations

```python
# Circuit analysis dashboard
fig = visualizer.plot_quantum_circuit_analysis(
    circuit_results=results_dict,
    circuit_info=circuit_information,
    title="Custom Analysis Dashboard"
)

# Trajectory evolution
fig = visualizer.plot_trajectory_evolution(
    trajectories=trajectory_array,
    time_steps=time_array,
    title="Quantum State Evolution"
)
```

## Configuration & Error Handling

The package includes comprehensive error handling and logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Robust quantum device initialization
qgb = QuantumGaltonBoard(
    n_layers=4, 
    n_shots=1000,
    device_name="default.qubit",  # or quantum hardware
    seed=42  # for reproducibility
)
```

### Common Configuration Options

```python
# Different quantum devices
qgb_cpu = QuantumGaltonBoard(device_name="default.qubit")
qgb_gpu = QuantumGaltonBoard(device_name="default.qubit.torch")

# AI model configuration
ai_model = QuantumGaltonAI(
    device='auto',      # or 'cpu', 'cuda'
    batch_size=64,
    lr=1e-3,
    beta=1.0           # KL divergence weighting
)

# Visualization themes
visualizer = PlinkoDynamicsVisualizer(theme='purple_navy')
```

## What's New in v2.0

- **Complete Modular Rewrite**: Clean, maintainable architecture  
- **Beautiful Purple/Navy Visualizations**: Professional color themes  
- **Comprehensive Error Handling**: Robust validation and logging  
- **Enhanced AI Models**: Improved training stability and performance  
- **Advanced Metrics**: 12+ statistical distance measures  
- **Publication Ready**: High-quality plots and documentation  
- **Hardware Compatible**: Ready for quantum devices

## Common Issues & Solutions

### Installation Issues
```bash
# Missing dependencies
pip install pennylane torch matplotlib seaborn scipy

# Optional AI features
pip install torchdiffeq

# Development tools
pip install pytest black isort
```

### Runtime Issues
```python
# Memory issues with large simulations
qgb = QuantumGaltonBoard(n_layers=3, n_shots=500)  # Reduce size

# GPU/CUDA issues
ai_model = QuantumGaltonAI(device='cpu')  # Force CPU usage

# Visualization backend issues
import matplotlib
matplotlib.use('Agg')  # For headless systems
```

## Contributing

We welcome contributions! Here's how to get started:

```bash
# Fork and clone the repository
git clone https://github.com/your-username/Plinko-Dynamics.git
cd Plinko-Dynamics

# Create development environment
pip install -e .
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Format code
black plinko_dynamics/
isort plinko_dynamics/

# Submit pull request
```

### Code Quality Standards
- **Type Hints**: All functions have comprehensive type annotations
- **Documentation**: Detailed docstrings following Google style
- **Error Handling**: Comprehensive validation and error messages
- **Testing**: Unit tests for all major components
- **Logging**: Structured logging throughout

## Examples

Check out these examples to get started:

- **Basic Tutorial**: `examples/basic_tutorial.py`
- **Advanced Simulations**: `examples/advanced_examples.py`
- **AI Integration**: `examples/ai_training.py`
- **Custom Visualizations**: `examples/visualization_gallery.py`
- **Metrics Analysis**: `examples/statistical_analysis.py`

## Future Roadmap

### Version 2.1 (Next Release)
- [ ] Real-time visualization dashboard
- [ ] Quantum hardware integration examples
- [ ] Parameter optimization algorithms
- [ ] Enhanced AI architectures

### Version 3.0 (Future Vision)
- [ ] Web-based interface
- [ ] Distributed computing support
- [ ] Advanced quantum error correction
- [ ] Machine learning circuit design

## Performance Benchmarks

### Quantum Simulation
- **Small circuits** (≤4 layers): ~0.1s per simulation
- **Medium circuits** (5-8 layers): ~0.5s per simulation
- **Large circuits** (9+ layers): ~2s per simulation

### AI Training
- **CPU**: ~30s per epoch (small datasets)
- **GPU**: ~5s per epoch (small datasets)

### Visualization
- **Static plots**: ~0.5s generation
- **Interactive dashboards**: ~2s generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Support & Contact

- **Bug Reports**: [GitHub Issues](https://github.com/its-serah/Plinko-Dynamics/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/its-serah/Plinko-Dynamics/discussions)
- **Questions**: Open an issue for support
- **Discussions**: Join our community discussions

## Recognition

This project demonstrates the elegant intersection of:
- **Quantum Computing** - Leveraging quantum superposition and interference
- **Classical Physics** - Understanding statistical mechanics through Galton boards
- **Artificial Intelligence** - Learning complex quantum dynamics
- **Data Science** - Advanced statistical analysis and visualization

---
  
  <p><strong>If you find this project helpful, please consider giving it a star!</strong></p>
</div>
