"""
Plinko Dynamics: Quantum Galton Board Simulation Package

A sophisticated quantum simulation package for modeling Plinko/Galton Board
dynamics using quantum circuits and AI-enhanced analysis.
"""

__version__ = "2.0.0"
__author__ = "Serah Rashidi"
__email__ = "serah@example.com"

from .core.quantum_galton import QuantumGaltonBoard
from .ai.models import QuantumGaltonAI
from .visualization.plotter import PlinkoDynamicsVisualizer
from .utils.metrics import DistributionMetrics

__all__ = [
    'QuantumGaltonBoard',
    'QuantumGaltonAI', 
    'PlinkoDynamicsVisualizer',
    'DistributionMetrics'
]
