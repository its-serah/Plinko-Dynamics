"""
Version information for Plinko Dynamics.

This file is managed by the version control system.
"""

__version__ = "2.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))
__author__ = "Serah Rashidi"
__email__ = "its-serah@github.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Serah Rashidi"
__url__ = "https://github.com/its-serah/Plinko-Dynamics"
__description__ = "Advanced Quantum Galton Board Simulation with AI-Enhanced Analysis"

# Feature flags
FEATURES = {
    "quantum_simulation": True,
    "ai_analysis": True,
    "advanced_visualization": True,
    "benchmarking": True,
    "hardware_support": False,  # Set to True when quantum hardware is available
    "gpu_acceleration": False,  # Set to True when CUDA is available
}

def get_version():
    """Return the current version string."""
    return __version__

def get_version_info():
    """Return version information as a dictionary."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
        "features": FEATURES,
    }
