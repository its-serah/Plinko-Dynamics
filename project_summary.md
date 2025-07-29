# Quantum Galton Board: Universal Statistical Simulator Implementation

## Project Overview

This project explores the implementation of quantum circuits to simulate complex statistical systems through a Galton Box-style Monte Carlo problem. The approach is based on the Universal Statistical Simulator concept, which demonstrates the potential for exponential speed-up over classical methods in high-dimensional statistical problems.

## Understanding of the Problem

### Classical Galton Board
The classical Galton board (or Plinko game) consists of a triangular arrangement of pegs where balls fall through, hitting pegs and being deflected left or right with equal probability. This creates a binomial distribution that approximates a Gaussian distribution for large numbers of layers.

### Quantum Implementation
The quantum version uses quantum circuits to simulate the same statistical behavior:

1. **Quantum State Representation**: Each possible path through the board is represented by a quantum state
2. **Superposition**: The quantum system can explore all possible paths simultaneously
3. **Measurement**: Final measurements collapse the superposition to obtain classical outcomes

### Circuit Structure
- **Control Qubits**: Determine the direction at each peg (left/right)
- **Position Qubits**: Track the current position of the ball
- **CSWAP Gates**: Implement the conditional movement based on control qubit states
- **Measurement**: Extract the final position distribution

## Technical Implementation

### Core Algorithm
1. Initialize ancilla qubits in specific states
2. For each layer:
   - Apply rotation gates to control qubits (determines bias)
   - Use CSWAP gates to conditionally move the position
   - Apply CNOT gates for entanglement
3. Measure final positions to obtain distribution

### Target Distributions
- **Gaussian Distribution**: Using standard rotation angles (π/4 variations)
- **Exponential Distribution**: Modified rotation angles based on exponential decay
- **Hadamard Quantum Walk**: Using Hadamard gates for equal probability

## Applications and Relevance

### Monte Carlo Methods
This quantum approach is relevant to:
- **Particle Transport**: Simulating particle scattering and diffusion
- **Financial Modeling**: Risk assessment and option pricing
- **Statistical Physics**: Thermodynamic system simulations
- **Machine Learning**: Sampling from complex probability distributions

### Quantum Advantage
The potential exponential speedup comes from:
- **Parallel Path Exploration**: Quantum superposition allows simultaneous exploration
- **Interference Effects**: Quantum interference can enhance or suppress certain outcomes
- **Entanglement**: Creates correlations not possible classically

## Challenges and Considerations

### Noise Effects
Real quantum hardware introduces:
- **Gate Errors**: Imperfect quantum operations
- **Decoherence**: Loss of quantum coherence over time
- **Measurement Errors**: Imperfect state readout

### Optimization Strategies
- **Circuit Depth Reduction**: Minimize number of sequential operations
- **Error Mitigation**: Techniques to reduce noise impact
- **Hardware-Specific Optimization**: Adapt to specific quantum device constraints

## Expected Outcomes

### Verification Metrics
- **Distribution Accuracy**: Compare quantum vs classical distributions
- **Statistical Distance Measures**: KL-divergence, total variation distance
- **Noise Tolerance**: Performance under realistic noise conditions
- **Scalability**: Performance as number of layers increases

### Success Criteria
- Quantum distributions match theoretical expectations within statistical error
- Different target distributions are successfully implemented
- Noise-resilient implementations show acceptable performance
- Clear demonstration of quantum circuit functionality

## Conclusion

This project demonstrates the practical implementation of quantum Monte Carlo methods using a well-understood classical system as a benchmark. The Galton board serves as an excellent testbed for exploring quantum advantages in statistical simulation while providing clear visualization and verification of results. The work contributes to the broader understanding of quantum algorithms for statistical computation and their potential applications in scientific computing and machine learning.
