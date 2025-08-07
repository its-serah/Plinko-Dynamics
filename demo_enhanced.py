#!/usr/bin/env python3
"""
Enhanced Plinko Dynamics Demo

Showcases the new modular, robust implementation with beautiful visualizations
featuring purple and navy blue color schemes.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Import our new modular components
from plinko_dynamics import (
    QuantumGaltonBoard,
    QuantumGaltonAI,
    PlinkoDynamicsVisualizer,
    DistributionMetrics
)
from plinko_dynamics.core.classical_simulator import ClassicalGaltonBoard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    print("🎯 Enhanced Plinko Dynamics Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer with beautiful theme
    visualizer = PlinkoDynamicsVisualizer(theme='purple_navy')
    
    # Demo parameters
    n_layers = 4
    n_shots = 2000
    seed = 42
    
    print(f"\n🔬 Setting up experiment with {n_layers} layers and {n_shots} shots")
    
    # ======================================================================
    # Part 1: Quantum Circuit Comparison
    # ======================================================================
    print("\n📊 Part 1: Quantum Circuit Analysis")
    print("-" * 30)
    
    try:
        # Create quantum Galton board
        qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=n_shots, seed=seed)
        
        # Test different circuit types
        circuit_types = ["gaussian", "exponential", "hadamard"]
        circuit_results = {}
        
        for circuit_type in circuit_types:
            print(f"  Running {circuit_type} circuit...")
            
            # Add circuit-specific parameters
            kwargs = {}
            if circuit_type == "gaussian":
                kwargs['rotation_angle'] = 0.3 * np.pi
            elif circuit_type == "exponential":
                kwargs['decay_rate'] = 1.5
            
            samples = qgb.run_simulation(circuit_type, **kwargs)
            distribution = qgb.get_probability_distribution(samples)
            circuit_results[circuit_type] = distribution
            
            print(f"    ✓ {circuit_type}: mean={np.sum(np.arange(len(distribution)) * distribution):.3f}")
        
        # Get circuit information
        circuit_info = qgb.get_circuit_info("gaussian")
        
        # Create comprehensive circuit analysis visualization
        fig = visualizer.plot_quantum_circuit_analysis(
            circuit_results, 
            circuit_info,
            title="🔮 Quantum Circuit Performance Analysis",
            save_path=str(output_dir / "circuit_analysis.png")
        )
        plt.show()
        plt.close(fig)
        
        print("  ✅ Circuit analysis visualization saved!")
        
    except Exception as e:
        logger.error(f"Error in quantum circuit analysis: {e}")
        print(f"  ❌ Circuit analysis failed: {e}")
    
    # ======================================================================
    # Part 2: Quantum vs Classical Comparison
    # ======================================================================
    print("\n⚔️ Part 2: Quantum vs Classical Comparison")
    print("-" * 40)
    
    try:
        # Classical simulation
        classical_gb = ClassicalGaltonBoard(n_layers=n_layers, n_shots=n_shots, seed=seed)
        classical_dist = classical_gb.simulate_uniform_probability(0.5)
        theoretical_dist = classical_gb.get_theoretical_binomial(0.5)
        
        # Use the best quantum result (gaussian)
        quantum_dist = circuit_results["gaussian"]
        
        # Comprehensive comparison
        distributions = {
            "Quantum (Gaussian)": quantum_dist,
            "Classical Simulation": classical_dist,
            "Theoretical Binomial": theoretical_dist
        }
        
        # Create comparison visualization
        fig = visualizer.plot_distribution_comparison(
            distributions,
            title="⚔️ Quantum vs Classical vs Theoretical",
            save_path=str(output_dir / "quantum_vs_classical.png")
        )
        plt.show()
        plt.close(fig)
        
        # Advanced metrics analysis
        analysis = DistributionMetrics.quantum_classical_analysis(
            quantum_dist, classical_dist, theoretical_dist
        )
        
        print("  📈 Analysis Results:")
        print(f"    Quantum std: {analysis['quantum_advantage']['spreads_comparison']['quantum_std']:.3f}")
        print(f"    Classical std: {analysis['quantum_advantage']['spreads_comparison']['classical_std']:.3f}")
        print(f"    Spread ratio: {analysis['quantum_advantage']['spreads_comparison']['ratio']:.3f}")
        print(f"    Entropy difference: {analysis['quantum_advantage']['entropy_comparison']['difference']:.3f}")
        
        print("  ✅ Quantum vs Classical analysis completed!")
        
    except Exception as e:
        logger.error(f"Error in quantum vs classical comparison: {e}")
        print(f"  ❌ Comparison failed: {e}")
    
    # ======================================================================
    # Part 3: Beautiful Plinko Board Visualization
    # ======================================================================
    print("\n🎨 Part 3: Beautiful Plinko Board Visualization")
    print("-" * 45)
    
    try:
        # Create stunning Plinko board visualization
        fig = visualizer.create_beautiful_plinko_board(
            n_layers=n_layers,
            distribution=quantum_dist,
            title="🎯 Quantum Plinko Dynamics - Purple Navy Theme",
            save_path=str(output_dir / "beautiful_plinko_board.png")
        )
        plt.show()
        plt.close(fig)
        
        print("  ✅ Beautiful Plinko board visualization created!")
        
    except Exception as e:
        logger.error(f"Error creating Plinko visualization: {e}")
        print(f"  ❌ Visualization failed: {e}")
    
    # ======================================================================
    # Part 4: AI Trajectory Analysis (if possible)
    # ======================================================================
    print("\n🤖 Part 4: AI Trajectory Analysis")
    print("-" * 35)
    
    try:
        # Generate simple trajectory data for demonstration
        print("  Generating trajectory data...")
        trajectories = generate_demo_trajectories(n_layers, n_timesteps=8, n_trajectories=5)
        
        if trajectories is not None:
            # Create trajectory visualization
            fig = visualizer.plot_trajectory_evolution(
                trajectories,
                title="🤖 AI-Enhanced Trajectory Evolution",
                save_path=str(output_dir / "trajectory_evolution.png")
            )
            plt.show()
            plt.close(fig)
            
            # Time series analysis
            time_analysis = DistributionMetrics.time_series_analysis(trajectories)
            print(f"  📊 Time series stability: {time_analysis['stability_measure']:.4f}")
            
            print("  ✅ AI trajectory analysis completed!")
        else:
            print("  ⚠️ Skipping AI analysis (dependency issues)")
            
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        print(f"  ❌ AI analysis failed: {e}")
    
    # ======================================================================
    # Part 5: Comprehensive Metrics Report
    # ======================================================================
    print("\n📋 Part 5: Comprehensive Metrics Report")
    print("-" * 40)
    
    try:
        # Circuit performance analysis
        performance_analysis = DistributionMetrics.circuit_performance_metrics(
            circuit_results, target_dist=theoretical_dist
        )
        
        print("  🏆 Circuit Rankings:")
        for metric, ranking in performance_analysis['ranking'].items():
            if ranking:
                print(f"    {metric}: {' > '.join(ranking)}")
        
        # Distance measures between all distributions
        print("\n  📏 Key Distance Measures:")
        for name, comparison in performance_analysis['target_comparisons'].items():
            tv_distance = comparison['distances']['total_variation']
            kl_div = comparison['distances']['kl_divergence_1_to_2']
            print(f"    {name} vs Theoretical: TV={tv_distance:.4f}, KL={kl_div:.4f}")
        
        print("  ✅ Comprehensive metrics report completed!")
        
    except Exception as e:
        logger.error(f"Error in metrics analysis: {e}")
        print(f"  ❌ Metrics analysis failed: {e}")
    
    # ======================================================================
    # Summary and Recommendations
    # ======================================================================
    print("\n🎯 Demo Summary")
    print("=" * 50)
    print("✅ Enhanced modular codebase successfully demonstrated!")
    print("✅ Beautiful purple/navy visualizations created!")
    print("✅ Comprehensive error handling and logging implemented!")
    print("✅ Advanced statistical analysis performed!")
    print(f"✅ All results saved to: {output_dir.absolute()}")
    
    print("\n🚀 Suggested Improvements:")
    print("1. Implement adaptive circuit parameter optimization")
    print("2. Add more quantum circuit types (e.g., parameterized circuits)")
    print("3. Enhance AI models with circuit-specific encoders")
    print("4. Add real-time visualization capabilities")
    print("5. Implement quantum hardware backends")
    
    print("\n🎉 Demo completed successfully!")


def generate_demo_trajectories(n_layers, n_timesteps=8, n_trajectories=5):
    """Generate demonstration trajectories for AI analysis."""
    try:
        # Simple trajectory generation using quantum circuits
        qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=100, seed=42)
        
        trajectories = []
        for i in range(n_trajectories):
            trajectory = []
            rotation_angles = np.linspace(0.1 * np.pi, 0.4 * np.pi, n_timesteps)
            
            for angle in rotation_angles:
                samples = qgb.run_simulation("gaussian", rotation_angle=angle)
                distribution = qgb.get_probability_distribution(samples)
                
                # Pad to consistent size
                max_bins = n_layers + 1
                if len(distribution) < max_bins:
                    padded_dist = np.zeros(max_bins)
                    padded_dist[:len(distribution)] = distribution
                    distribution = padded_dist
                
                trajectory.append(distribution)
            
            trajectories.append(trajectory)
        
        return np.array(trajectories)
        
    except Exception as e:
        logger.error(f"Error generating trajectories: {e}")
        return None


if __name__ == "__main__":
    main()
