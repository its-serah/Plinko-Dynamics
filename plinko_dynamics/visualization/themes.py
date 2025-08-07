"""
Beautiful Color Themes for Plinko Dynamics Visualizations

This module provides stunning color schemes with purple and navy blue themes
for creating professional and aesthetically pleasing visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


class PurpleNavyTheme:
    """
    Beautiful purple and navy blue color theme for Plinko Dynamics visualizations.
    """
    
    # Primary color palette
    DEEP_NAVY = '#1B2951'
    ROYAL_BLUE = '#2E4BC6'
    ELECTRIC_PURPLE = '#6B46C1'
    LAVENDER_PURPLE = '#A855F7'
    SOFT_PURPLE = '#C084FC'
    LIGHT_PURPLE = '#DDD6FE'
    PEARL_WHITE = '#F8FAFC'
    
    # Secondary accent colors
    TEAL_ACCENT = '#14B8A6'
    PINK_ACCENT = '#EC4899'
    GOLD_ACCENT = '#F59E0B'
    
    # Gradient colors for plots
    GRADIENT_COLORS = [DEEP_NAVY, ROYAL_BLUE, ELECTRIC_PURPLE, LAVENDER_PURPLE, SOFT_PURPLE]
    
    @classmethod
    def setup_style(cls):
        """Set up the matplotlib style with purple-navy theme."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Custom style parameters
        custom_params = {
            'figure.facecolor': cls.PEARL_WHITE,
            'axes.facecolor': cls.PEARL_WHITE,
            'axes.edgecolor': cls.DEEP_NAVY,
            'axes.linewidth': 1.2,
            'axes.labelcolor': cls.DEEP_NAVY,
            'axes.titlecolor': cls.DEEP_NAVY,
            'axes.titleweight': 'bold',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.color': cls.DEEP_NAVY,
            'ytick.color': cls.DEEP_NAVY,
            'text.color': cls.DEEP_NAVY,
            'grid.color': cls.LIGHT_PURPLE,
            'grid.alpha': 0.3,
            'legend.facecolor': cls.PEARL_WHITE,
            'legend.edgecolor': cls.DEEP_NAVY,
            'legend.framealpha': 0.9,
        }
        
        plt.rcParams.update(custom_params)
    
    @classmethod
    def get_color_palette(cls, n_colors=5):
        """
        Get a color palette with n_colors.
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of hex color codes
        """
        if n_colors <= len(cls.GRADIENT_COLORS):
            return cls.GRADIENT_COLORS[:n_colors]
        
        # Create interpolated colors for more than 5 colors
        colors = []
        for i in range(n_colors):
            ratio = i / (n_colors - 1) if n_colors > 1 else 0
            if ratio <= 0.25:
                # Deep navy to royal blue
                t = ratio * 4
                colors.append(cls._interpolate_color(cls.DEEP_NAVY, cls.ROYAL_BLUE, t))
            elif ratio <= 0.5:
                # Royal blue to electric purple
                t = (ratio - 0.25) * 4
                colors.append(cls._interpolate_color(cls.ROYAL_BLUE, cls.ELECTRIC_PURPLE, t))
            elif ratio <= 0.75:
                # Electric purple to lavender purple
                t = (ratio - 0.5) * 4
                colors.append(cls._interpolate_color(cls.ELECTRIC_PURPLE, cls.LAVENDER_PURPLE, t))
            else:
                # Lavender purple to soft purple
                t = (ratio - 0.75) * 4
                colors.append(cls._interpolate_color(cls.LAVENDER_PURPLE, cls.SOFT_PURPLE, t))
        
        return colors
    
    @classmethod
    def _interpolate_color(cls, color1, color2, t):
        """Interpolate between two hex colors."""
        # Convert hex to RGB
        rgb1 = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
        rgb2 = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
        
        # Interpolate
        rgb = tuple(int(rgb1[i] + t * (rgb2[i] - rgb1[i])) for i in range(3))
        
        # Convert back to hex
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    @classmethod
    def create_custom_colormap(cls, name='purple_navy'):
        """
        Create a custom colormap with purple-navy gradient.
        
        Args:
            name: Name for the colormap
            
        Returns:
            LinearSegmentedColormap object
        """
        colors = [cls.DEEP_NAVY, cls.ROYAL_BLUE, cls.ELECTRIC_PURPLE, 
                 cls.LAVENDER_PURPLE, cls.SOFT_PURPLE, cls.LIGHT_PURPLE]
        
        return LinearSegmentedColormap.from_list(name, colors, N=256)
    
    @classmethod
    def get_quantum_classical_colors(cls):
        """Get specific colors for quantum vs classical comparisons."""
        return {
            'quantum': cls.ELECTRIC_PURPLE,
            'classical': cls.ROYAL_BLUE,
            'theoretical': cls.TEAL_ACCENT,
            'ai_prediction': cls.PINK_ACCENT,
            'error': cls.GOLD_ACCENT
        }
    
    @classmethod
    def get_circuit_colors(cls):
        """Get colors for different circuit types."""
        return {
            'gaussian': cls.ELECTRIC_PURPLE,
            'exponential': cls.LAVENDER_PURPLE,
            'hadamard': cls.ROYAL_BLUE,
            'uniform': cls.TEAL_ACCENT
        }


class QuantumCircuitTheme:
    """Theme specifically for quantum circuit visualizations."""
    
    GATE_COLORS = {
        'H': PurpleNavyTheme.ELECTRIC_PURPLE,
        'RX': PurpleNavyTheme.LAVENDER_PURPLE,
        'CNOT': PurpleNavyTheme.ROYAL_BLUE,
        'CSWAP': PurpleNavyTheme.DEEP_NAVY,
        'MEASURE': PurpleNavyTheme.TEAL_ACCENT,
        'WIRE': PurpleNavyTheme.DEEP_NAVY
    }
    
    @classmethod
    def setup_circuit_style(cls):
        """Setup style parameters for circuit diagrams."""
        return {
            'wire_color': cls.GATE_COLORS['WIRE'],
            'gate_color': cls.GATE_COLORS['H'],
            'text_color': PurpleNavyTheme.PEARL_WHITE,
            'background_color': PurpleNavyTheme.PEARL_WHITE
        }
