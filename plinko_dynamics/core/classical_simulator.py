"""
Classical Galton Board Simulator

A clean implementation of the classical Galton board for comparison
with quantum results and validation purposes.
"""

import numpy as np
from typing import List, Optional
import random


class ClassicalGaltonBoard:
    """
    Classical implementation of the Galton Board for comparison with quantum results.
    
    This simulator provides various probability models for the classical
    Galton board behavior.
    """
    
    def __init__(self, n_layers: int, n_shots: int = 1000, seed: Optional[int] = None):
        """
        Initialize the classical Galton board.
        
        Args:
            n_layers: Number of layers in the Galton board
            n_shots: Number of balls to simulate
            seed: Random seed for reproducibility
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_shots <= 0:
            raise ValueError(f"n_shots must be > 0, got {n_shots}")
            
        self.n_layers = n_layers
        self.n_shots = n_shots
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def simulate_uniform_probability(self, probability: float = 0.5) -> np.ndarray:
        """
        Simulate classical Galton board with uniform probability at each peg.
        
        Args:
            probability: Probability of going right at each peg (0.5 for unbiased)
            
        Returns:
            Probability distribution array
        """
        if not (0 <= probability <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
            
        containers = np.zeros(self.n_layers + 1)
        
        for _ in range(self.n_shots):
            position = 0
            for _ in range(self.n_layers):
                if random.random() < probability:
                    position += 1
            containers[position] += 1
        
        return containers / self.n_shots
    
    def simulate_variable_probability(self, probabilities: List[float]) -> np.ndarray:
        """
        Simulate with different probabilities for each layer.
        
        Args:
            probabilities: List of probabilities for each layer
            
        Returns:
            Probability distribution array
        """
        if len(probabilities) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} probabilities, got {len(probabilities)}")
        
        if not all(0 <= p <= 1 for p in probabilities):
            raise ValueError("All probabilities must be between 0 and 1")
            
        containers = np.zeros(self.n_layers + 1)
        
        for _ in range(self.n_shots):
            position = 0
            for layer_prob in probabilities:
                if random.random() < layer_prob:
                    position += 1
            containers[position] += 1
        
        return containers / self.n_shots
    
    def simulate_biased_walk(self, bias: float = 0.0) -> np.ndarray:
        """
        Simulate a biased random walk.
        
        Args:
            bias: Bias parameter (-1 = always left, 0 = unbiased, 1 = always right)
            
        Returns:
            Probability distribution array
        """
        if not (-1 <= bias <= 1):
            raise ValueError(f"Bias must be between -1 and 1, got {bias}")
            
        # Convert bias to probability
        probability = (bias + 1) / 2
        return self.simulate_uniform_probability(probability)
    
    def get_theoretical_binomial(self, probability: float = 0.5) -> np.ndarray:
        """
        Get the theoretical binomial distribution for comparison.
        
        Args:
            probability: Probability parameter for binomial distribution
            
        Returns:
            Theoretical probability distribution
        """
        from scipy.stats import binom
        
        x = np.arange(self.n_layers + 1)
        return binom.pmf(x, self.n_layers, probability)
    
    def get_statistics(self, distribution: np.ndarray) -> dict:
        """
        Calculate statistics for a given distribution.
        
        Args:
            distribution: Probability distribution array
            
        Returns:
            Dictionary containing statistical measures
        """
        x = np.arange(len(distribution))
        
        mean = np.sum(x * distribution)
        variance = np.sum((x - mean) ** 2 * distribution)
        std_dev = np.sqrt(variance)
        
        # Skewness and kurtosis
        skewness = np.sum(((x - mean) / std_dev) ** 3 * distribution) if std_dev > 0 else 0
        kurtosis = np.sum(((x - mean) / std_dev) ** 4 * distribution) if std_dev > 0 else 0
        
        return {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def compare_distributions(self, dist1: np.ndarray, dist2: np.ndarray) -> dict:
        """
        Compare two probability distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Dictionary containing comparison metrics
        """
        if len(dist1) != len(dist2):
            raise ValueError("Distributions must have the same length")
        
        # Mean Squared Error
        mse = np.mean((dist1 - dist2) ** 2)
        
        # Kullback-Leibler Divergence (with small epsilon to avoid log(0))
        eps = 1e-10
        kl_div = np.sum(dist1 * np.log((dist1 + eps) / (dist2 + eps)))
        
        # Total Variation Distance
        tv_distance = 0.5 * np.sum(np.abs(dist1 - dist2))
        
        # Chi-squared distance
        chi_squared = np.sum((dist1 - dist2) ** 2 / (dist2 + eps))
        
        # Bhattacharyya distance
        bhattacharyya = -np.log(np.sum(np.sqrt(dist1 * dist2)))
        
        return {
            'mse': mse,
            'kl_divergence': kl_div,
            'tv_distance': tv_distance,
            'chi_squared': chi_squared,
            'bhattacharyya': bhattacharyya
        }
