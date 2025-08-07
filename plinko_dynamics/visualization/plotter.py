"""
Advanced Visualization Module for Plinko Dynamics

Beautiful, publication-ready visualizations with purple and navy blue themes
for quantum Galton board simulations and AI analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
import warnings

from .themes import PurpleNavyTheme, QuantumCircuitTheme


class PlinkoDynamicsVisualizer:
    """
    Advanced visualization class for Plinko Dynamics with stunning visuals.
    """
    
    def __init__(self, theme: str = 'purple_navy', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            theme: Theme name ('purple_navy' or 'quantum_circuit')
            figsize: Default figure size
        """
        self.theme = theme
        self.figsize = figsize
        
        # Setup theme
        if theme == 'purple_navy':
            PurpleNavyTheme.setup_style()
            self.colors = PurpleNavyTheme.get_quantum_classical_colors()
            self.circuit_colors = PurpleNavyTheme.get_circuit_colors()
        else:
            PurpleNavyTheme.setup_style()  # Default fallback
            self.colors = PurpleNavyTheme.get_quantum_classical_colors()
            self.circuit_colors = PurpleNavyTheme.get_circuit_colors()
    
    def plot_distribution_comparison(self, 
                                   distributions: Dict[str, np.ndarray],
                                   title: str = "Quantum vs Classical Distributions",
                                   save_path: Optional[str] = None,
                                   show_statistics: bool = True) -> plt.Figure:
        """
        Create a beautiful comparison plot of multiple distributions.
        
        Args:
            distributions: Dictionary with labels as keys and distributions as values
            title: Plot title
            save_path: Path to save the figure (optional)
            show_statistics: Whether to show statistical information
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Main comparison plot
        ax1 = axes[0, 0]
        x = np.arange(len(next(iter(distributions.values()))))
        
        for i, (label, dist) in enumerate(distributions.items()):
            color = self._get_color_for_label(label)
            ax1.bar(x + i*0.15, dist, width=0.15, label=label, color=color, alpha=0.8)
        
        ax1.set_xlabel('Bin Position')
        ax1.set_ylabel('Probability')
        ax1.set_title('Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Line plot comparison
        ax2 = axes[0, 1]
        for label, dist in distributions.items():
            color = self._get_color_for_label(label)
            ax2.plot(x, dist, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Bin Position')
        ax2.set_ylabel('Probability')
        ax2.set_title('Line Plot Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Heatmap
        ax3 = axes[1, 0]
        dist_matrix = np.array([dist for dist in distributions.values()])
        cmap = PurpleNavyTheme.create_custom_colormap()
        im = ax3.imshow(dist_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
        ax3.set_yticks(range(len(distributions)))
        ax3.set_yticklabels(distributions.keys())
        ax3.set_xlabel('Bin Position')
        ax3.set_title('Distribution Heatmap')
        plt.colorbar(im, ax=ax3, label='Probability')
        
        # Statistical summary
        ax4 = axes[1, 1]
        if show_statistics:
            stats_text = self._compute_distribution_stats(distributions)
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=PurpleNavyTheme.LIGHT_PURPLE, alpha=0.8))
        ax4.set_title('Statistical Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=PurpleNavyTheme.PEARL_WHITE)
        
        return fig
    
    def plot_trajectory_evolution(self,
                                trajectories: np.ndarray,
                                time_steps: Optional[np.ndarray] = None,
                                title: str = "Quantum Trajectory Evolution",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the evolution of quantum trajectories over time.
        
        Args:
            trajectories: Array of shape (n_trajectories, n_timesteps, n_bins)
            time_steps: Time step array (optional)
            title: Plot title
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        if time_steps is None:
            time_steps = np.arange(trajectories.shape[1])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Individual trajectory plots
        ax1 = axes[0, 0]
        colors = PurpleNavyTheme.get_color_palette(min(5, trajectories.shape[0]))
        
        for i in range(min(3, trajectories.shape[0])):
            for t in range(trajectories.shape[1]):
                ax1.plot(trajectories[i, t], alpha=0.6, color=colors[i], linewidth=1.5)
        ax1.set_title('Individual Trajectories')
        ax1.set_xlabel('Bin Position')
        ax1.set_ylabel('Probability')
        ax1.grid(True, alpha=0.3)
        
        # Average trajectory evolution
        ax2 = axes[0, 1]
        avg_trajectory = trajectories.mean(axis=0)
        cmap = PurpleNavyTheme.create_custom_colormap()
        im2 = ax2.imshow(avg_trajectory.T, aspect='auto', cmap=cmap, origin='lower')
        ax2.set_title('Average Trajectory Evolution')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Bin Position')
        plt.colorbar(im2, ax=ax2)
        
        # Variance across trajectories
        ax3 = axes[0, 2]
        var_trajectory = trajectories.var(axis=0)
        im3 = ax3.imshow(var_trajectory.T, aspect='auto', cmap='plasma', origin='lower')
        ax3.set_title('Variance Across Trajectories')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Bin Position')
        plt.colorbar(im3, ax=ax3)
        
        # Time evolution of specific bins
        ax4 = axes[1, 0]
        n_bins = trajectories.shape[2]
        interesting_bins = [0, n_bins//2, n_bins-1]  # First, middle, last
        
        for bin_idx in interesting_bins:
            bin_evolution = trajectories[:, :, bin_idx].mean(axis=0)
            ax4.plot(time_steps, bin_evolution, 'o-', 
                    label=f'Bin {bin_idx}', linewidth=2, markersize=4)
        ax4.set_title('Bin Evolution Over Time')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Average Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Distribution spread over time
        ax5 = axes[1, 1]
        spreads = []
        for t in range(trajectories.shape[1]):
            # Calculate standard deviation of each trajectory at time t
            std_devs = []
            for i in range(trajectories.shape[0]):
                bins = np.arange(trajectories.shape[2])
                mean_pos = np.sum(bins * trajectories[i, t])
                std_dev = np.sqrt(np.sum((bins - mean_pos)**2 * trajectories[i, t]))
                std_devs.append(std_dev)
            spreads.append(np.mean(std_devs))
        
        ax5.plot(time_steps, spreads, 'o-', color=PurpleNavyTheme.ELECTRIC_PURPLE, 
                linewidth=2, markersize=6)
        ax5.set_title('Distribution Spread vs Time')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Average Standard Deviation')
        ax5.grid(True, alpha=0.3)
        
        # Final distribution comparison
        ax6 = axes[1, 2]
        final_dists = trajectories[:, -1, :]
        avg_final = final_dists.mean(axis=0)
        std_final = final_dists.std(axis=0)
        
        x = np.arange(len(avg_final))
        ax6.bar(x, avg_final, yerr=std_final, capsize=5, 
               color=PurpleNavyTheme.ELECTRIC_PURPLE, alpha=0.7)
        ax6.set_title('Final Distribution (with std)')
        ax6.set_xlabel('Bin Position')
        ax6.set_ylabel('Probability')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=PurpleNavyTheme.PEARL_WHITE)
        
        return fig
    
    def plot_quantum_circuit_analysis(self,
                                    circuit_results: Dict[str, np.ndarray],
                                    circuit_info: Optional[Dict] = None,
                                    title: str = "Quantum Circuit Analysis",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive analysis plots for quantum circuits.
        
        Args:
            circuit_results: Dictionary with circuit types as keys and results as values
            circuit_info: Additional circuit information
            title: Plot title
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Circuit comparison
        ax1 = axes[0, 0]
        x = np.arange(len(next(iter(circuit_results.values()))))
        
        for circuit_type, distribution in circuit_results.items():
            color = self.circuit_colors.get(circuit_type, PurpleNavyTheme.ELECTRIC_PURPLE)
            ax1.plot(x, distribution, 'o-', label=circuit_type.title(), 
                    color=color, linewidth=2.5, markersize=6)
        
        ax1.set_xlabel('Bin Position')
        ax1.set_ylabel('Probability')
        ax1.set_title('Circuit Type Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heatmap of all circuits
        ax2 = axes[0, 1]
        circuit_matrix = np.array([dist for dist in circuit_results.values()])
        cmap = PurpleNavyTheme.create_custom_colormap()
        im = ax2.imshow(circuit_matrix, aspect='auto', cmap=cmap)
        ax2.set_yticks(range(len(circuit_results)))
        ax2.set_yticklabels([ct.title() for ct in circuit_results.keys()])
        ax2.set_xlabel('Bin Position')
        ax2.set_title('Circuit Results Heatmap')
        plt.colorbar(im, ax=ax2)
        
        # Statistical comparison radar chart
        ax3 = axes[1, 1]
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        self._create_statistics_radar(ax3, circuit_results)
        ax3.set_title('Statistical Comparison')
        
        # Circuit information display
        ax4 = axes[1, 1]
        if circuit_info:
            info_text = self._format_circuit_info(circuit_info)
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=PurpleNavyTheme.LIGHT_PURPLE, alpha=0.8))
        ax4.set_title('Circuit Information')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=PurpleNavyTheme.PEARL_WHITE)
        
        return fig
    
    def create_beautiful_plinko_board(self,
                                    n_layers: int,
                                    distribution: np.ndarray,
                                    title: str = "Quantum Plinko Dynamics",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a beautiful visualization of the Plinko board with results.
        
        Args:
            n_layers: Number of layers in the board
            distribution: Probability distribution to visualize
            title: Plot title
            save_path: Save path (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Draw the Plinko board
        ax1.set_xlim(-0.5, n_layers + 0.5)
        ax1.set_ylim(-1, n_layers + 1)
        
        # Draw pegs
        for layer in range(n_layers):
            for peg in range(layer + 1):
                x = peg - layer/2 + n_layers/2
                y = n_layers - layer - 1
                
                # Peg visualization
                circle = plt.Circle((x, y), 0.1, 
                                  color=PurpleNavyTheme.DEEP_NAVY, alpha=0.8)
                ax1.add_patch(circle)
                
                # Draw paths (lines showing possible ball trajectories)
                if layer < n_layers - 1:
                    # Left path
                    ax1.plot([x, x - 0.5], [y, y - 1], 
                            color=PurpleNavyTheme.LIGHT_PURPLE, alpha=0.3, linewidth=1)
                    # Right path
                    ax1.plot([x, x + 0.5], [y, y - 1], 
                            color=PurpleNavyTheme.LIGHT_PURPLE, alpha=0.3, linewidth=1)
        
        # Draw collection bins
        bin_width = 0.8
        for i, prob in enumerate(distribution):
            x_bin = i - len(distribution)/2 + n_layers/2 + 0.5
            
            # Bin height proportional to probability
            height = prob * 3  # Scale for visibility
            
            # Color based on probability
            color_intensity = prob / max(distribution) if max(distribution) > 0 else 0
            color = plt.cm.get_cmap(PurpleNavyTheme.create_custom_colormap())(color_intensity)
            
            rect = FancyBboxPatch((x_bin - bin_width/2, -1), bin_width, height,
                                boxstyle="round,pad=0.02", 
                                facecolor=color, alpha=0.8,
                                edgecolor=PurpleNavyTheme.DEEP_NAVY, linewidth=1.5)
            ax1.add_patch(rect)
            
            # Add probability labels
            ax1.text(x_bin, height/2 - 0.8, f'{prob:.3f}', 
                    ha='center', va='center', fontsize=9, 
                    color=PurpleNavyTheme.DEEP_NAVY, fontweight='bold')
        
        ax1.set_title('Quantum Plinko Board Visualization')
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Distribution bar plot
        x = np.arange(len(distribution))
        colors = PurpleNavyTheme.get_color_palette(len(distribution))
        
        bars = ax2.bar(x, distribution, color=colors, alpha=0.8, 
                      edgecolor=PurpleNavyTheme.DEEP_NAVY, linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, distribution)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Bin Position')
        ax2.set_ylabel('Probability')
        ax2.set_title('Probability Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Statistics:
Mean: {np.sum(x * distribution):.3f}
Std: {np.sqrt(np.sum((x - np.sum(x * distribution))**2 * distribution)):.3f}
Max: {max(distribution):.3f}"""
        
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=PurpleNavyTheme.LIGHT_PURPLE, alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=PurpleNavyTheme.PEARL_WHITE)
        
        return fig
    
    def _get_color_for_label(self, label: str) -> str:
        """Get appropriate color for a given label."""
        label_lower = label.lower()
        
        if 'quantum' in label_lower:
            return self.colors['quantum']
        elif 'classical' in label_lower:
            return self.colors['classical']
        elif 'theoretical' in label_lower:
            return self.colors['theoretical']
        elif 'ai' in label_lower or 'prediction' in label_lower:
            return self.colors['ai_prediction']
        else:
            return PurpleNavyTheme.ELECTRIC_PURPLE
    
    def _compute_distribution_stats(self, distributions: Dict[str, np.ndarray]) -> str:
        """Compute statistical summary for distributions."""
        stats_text = "Statistical Summary:\n\n"
        
        for label, dist in distributions.items():
            x = np.arange(len(dist))
            mean = np.sum(x * dist)
            variance = np.sum((x - mean)**2 * dist)
            std_dev = np.sqrt(variance)
            
            stats_text += f"{label}:\n"
            stats_text += f"  Mean: {mean:.3f}\n"
            stats_text += f"  Std: {std_dev:.3f}\n"
            stats_text += f"  Max: {max(dist):.3f}\n\n"
        
        return stats_text
    
    def _create_statistics_radar(self, ax, distributions: Dict[str, np.ndarray]):
        """Create a radar chart for statistical comparison."""
        categories = ['Mean', 'Std Dev', 'Skewness', 'Max Prob']
        n_cats = len(categories)
        
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        colors = PurpleNavyTheme.get_color_palette(len(distributions))
        
        for i, (label, dist) in enumerate(distributions.items()):
            x = np.arange(len(dist))
            mean = np.sum(x * dist)
            std_dev = np.sqrt(np.sum((x - mean)**2 * dist))
            skewness = np.sum(((x - mean) / std_dev)**3 * dist) if std_dev > 0 else 0
            max_prob = max(dist)
            
            # Normalize values for radar chart
            values = [mean / len(dist), std_dev / 2, abs(skewness), max_prob]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True)
    
    def _format_circuit_info(self, info: Dict) -> str:
        """Format circuit information for display."""
        info_text = "Circuit Information:\n\n"
        
        important_keys = ['n_layers', 'n_shots', 'n_wires', 'circuit_type', 'depth']
        
        for key in important_keys:
            if key in info:
                formatted_key = key.replace('_', ' ').title()
                info_text += f"{formatted_key}: {info[key]}\n"
        
        return info_text
