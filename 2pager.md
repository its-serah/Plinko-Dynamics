# Quantum Walks & Monte Carlo Project
**WISER 2025 x NNL**  

## Quantum Galton Board Theory and Implementation: A Comprehensive Analysis

**Authors:** Elena, Krishang, Sarah  
**Date:** 2025-07-20  
**Team:** Plinko Dynamics  

---

### From Classical to Quantum: Core Mechanism
The classical Galton board demonstrates statistical mechanics through balls bouncing left or right through a triangular peg array, producing a Gaussian distribution. The quantum version transforms this concept by mapping each possible trajectory onto quantum computational basis states, enabling simultaneous calculation of all \(2^n\) trajectories through quantum superposition.

The Universal Statistical Simulator by Carney and Varcoe [1] achieves this using only three types of quantum gates with \(O(n^2)\) resource scaling compared to classical \(O(2^n)\) requirements. This exponential speedup emerges not from faster computation, but from a fundamentally different computational mechanism where probability amplitudes naturally interfere to produce the final distribution. Instead of enumerating each of the \(2^n\) possible trajectories separately, quantum computers encode all trajectories simultaneously as probability amplitudes in superposition. These amplitudes naturally interfere, constructively and destructively, automatically computing the final distribution through amplitude interference and measurement collapse rather than explicit trajectory summation.

---

### Quantum Implementation Mechanisms
Multiple research approaches reveal complementary implementation strategies. The gate-based approach from the Universal Statistical Simulator uses Hadamard gates to create equal superposition states at each peg, CNOT gates to entangle trajectory information between qubits, and rotation gates to adjust left-right probabilities for different distributions [1]. Each layer of the board corresponds to quantum operations that systematically evolve probability amplitudes, with the circuit depth scaling linearly with board layers rather than exponentially with trajectory count.

Photonic implementations using directional coupler matrices achieve similar quantum advantage through photon interference [2], functioning as simplified Boson Samplers that become exponentially hard to simulate classically.

The quantum walk perspective, as reviewed by Wang et al. [3], positions Galton boards within the broader framework of discrete-time quantum walks on graph structures. The triangular peg network represents a specific graph topology where quantum walkers explore multiple paths simultaneously, creating interference patterns that determine final position distributions.

---

### Universal Statistical Simulator Extension
The breakthrough insight lies in the system's extensibility. By removing pegs and altering left-right probability ratios, the basic Galton board becomes a universal statistical simulator capable of generating diverse probability distributions [1]. This modification principle directly enables exponential distributions through asymmetric peg biasing and Hadamard quantum walks through different circuit architectures, transforming the specific Galton structure into a flexible computational platform.

Experimental implementations on trapped-ion quantum computers demonstrate practical feasibility [4], showing how quantum walks and cellular automata principles translate to real hardware with programmable gate sequences and measured quantum states.

---

### Team Implementation Strategy
Our approach addresses all five project tasks through coordinated implementation. Task 2 develops a general algorithm for implementing quantum Galton boards with arbitrary layer counts, ensuring proper Gaussian distribution verification [1], while task 3 modifies this for exponential distributions [2] and Hadamard quantum walks [3]. Task 4 implements noise optimization for real quantum hardware through error mitigation techniques including Zero Noise Extrapolation and circuit-level optimizations. Task 5 provides quantitative validation by measuring discrepancies between experimental results and theoretical expectations while incorporating statistical error analysis.

Our implementation focuses on developing noise-resistant quantum circuits for statistical distribution generation, specifically targeting exponential and quantum walk patterns for solving partial differential equations in multi-dimensional systems with intricate coupling effects such as particle transport and quantum systems [3,4]. This directly supports NNL's simulation requirements while developing noise mitigation techniques applicable to near-term quantum devices, positioning the work within quantum computing's practical applications landscape.

---

### References
[1] Carney, M. & Varcoe, B. (2022). *Universal Statistical Simulator*. arXiv:2202.01735 [quant-ph]  
[2] Montanaro, A. (2015). *Quantum speedup of Monte Carlo methods*. arXiv:1504.06987 [quant-ph]  
[3] Wang, J. et al. (2024). *Review on Quantum Walk Computing: Theory, Implementation, and Application*. arXiv:2404.04178 [quant-ph]  
[4] Omanakuttan, G. et al. (2020). *Quantum walks and Dirac cellular automata on a programmable trapped-ion quantum computer*. *Nature Communications*, 11, 3720
