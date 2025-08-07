"""
Data Generation Utilities

Utilities for generating datasets for training and testing
quantum Galton board models.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional


class DatasetGenerator:
    """Generate datasets for quantum Galton board analysis."""
    
    @staticmethod
    def generate_simple_trajectories(n_layers: int = 4,
                                   n_trajectories: int = 10, 
                                   n_timesteps: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simple trajectory datasets.
        
        Args:
            n_layers: Number of layers in Galton board
            n_trajectories: Number of trajectory samples
            n_timesteps: Number of time steps per trajectory
            
        Returns:
            Tuple of (trajectories, parameters)
        """
        # This is a placeholder implementation
        # In practice, you would use the QuantumGaltonBoard class
        
        max_bins = n_layers + 1
        trajectories = []
        parameters = []
        
        for i in range(n_trajectories):
            trajectory = []
            # Generate varying parameters
            angle_start = 0.1 * np.pi + i * 0.05
            angle_end = 0.4 * np.pi + i * 0.03
            
            angles = np.linspace(angle_start, angle_end, n_timesteps)
            
            for t, angle in enumerate(angles):
                # Simple binomial-like distribution that varies with angle
                x = np.arange(max_bins)
                mean = (max_bins - 1) * angle / (0.5 * np.pi)
                std = np.sqrt(max_bins * 0.25)
                
                # Generate distribution
                dist = np.exp(-(x - mean)**2 / (2 * std**2))
                dist = dist / np.sum(dist)
                
                trajectory.append(dist)
            
            trajectories.append(trajectory)
            parameters.append([n_layers, angle_start, angle_end])
        
        return np.array(trajectories), np.array(parameters)
