"""
Plinko Dynamics: Quantum Galton Board Simulation Package

A sophisticated quantum simulation package for modeling Plinko/Galton Board
dynamics using quantum circuits and AI-enhanced analysis.
"""

# Version and author information
from ._version import (
    __version__,
    __version_info__,
    __author__,
    __email__,
    __license__,
    __copyright__,
    __url__,
    get_version,
    get_version_info,
)

# Core modules
from .core.quantum_galton import QuantumGaltonBoard
from .ai.models import QuantumGaltonAI
from .visualization.plotter import PlinkoDynamicsVisualizer
from .utils.metrics import DistributionMetrics
from .utils.benchmarks import PerformanceBenchmark, timeit, profile_memory

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # Core
    "QuantumGaltonBoard",
    "QuantumGaltonAI",
    "PlinkoDynamicsVisualizer",
    
    # Utilities
    "DistributionMetrics",
    "PerformanceBenchmark",
    "timeit",
    "profile_memory",
    
    # Versioning
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    "__url__",
    "get_version",
    "get_version_info",
]
