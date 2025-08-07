#!/usr/bin/env python3
"""
Quick Test Script for Plinko Dynamics v2.0

Tests the new modular architecture and key functionality.
"""

import numpy as np
import matplotlib.pyplot as plt

print("🧪 Testing Plinko Dynamics v2.0...")
print("=" * 40)

try:
    # Test imports
    print("📦 Testing imports...")
    from plinko_dynamics import (
        QuantumGaltonBoard,
        PlinkoDynamicsVisualizer, 
        DistributionMetrics
    )
    from plinko_dynamics.core.classical_simulator import ClassicalGaltonBoard
    print("✅ All imports successful!")
    
    # Test quantum simulation
    print("\n🔮 Testing quantum simulation...")
    qgb = QuantumGaltonBoard(n_layers=3, n_shots=500, seed=42)
    samples = qgb.run_simulation("gaussian")
    quantum_dist = qgb.get_probability_distribution(samples)
    print(f"✅ Quantum simulation: {quantum_dist}")
    
    # Test classical simulation
    print("\n⚖️ Testing classical simulation...")
    cgb = ClassicalGaltonBoard(n_layers=3, n_shots=500, seed=42)
    classical_dist = cgb.simulate_uniform_probability(0.5)
    print(f"✅ Classical simulation: {classical_dist}")
    
    # Test metrics
    print("\n📊 Testing metrics...")
    analysis = DistributionMetrics.quantum_classical_analysis(
        quantum_dist, classical_dist
    )
    tv_distance = analysis['basic_comparison']['distances']['total_variation']
    print(f"✅ Total variation distance: {tv_distance:.4f}")
    
    # Test visualization
    print("\n🎨 Testing visualization...")
    visualizer = PlinkoDynamicsVisualizer(theme='purple_navy')
    plt.ioff()  # Turn off interactive plotting
    fig = visualizer.create_beautiful_plinko_board(
        n_layers=3,
        distribution=quantum_dist,
        title="Test Visualization"
    )
    plt.close(fig)  # Close without showing
    print("✅ Visualization created successfully!")
    
    # Test circuit info
    print("\n🔧 Testing circuit information...")
    circuit_info = qgb.get_circuit_info("gaussian")
    print(f"✅ Circuit has {circuit_info['n_wires']} wires and {circuit_info['n_layers']} layers")
    
    print("\n🎯 All tests passed! Plinko Dynamics v2.0 is working correctly!")
    print("\n🚀 Suggestions for next steps:")
    print("1. Run 'python3 demo_enhanced.py' for full demonstration")
    print("2. Explore the modular package structure in plinko_dynamics/")
    print("3. Check out the beautiful purple/navy visualizations")
    print("4. Try different circuit types: gaussian, exponential, hadamard")
    print("5. Experiment with AI models (requires torchdiffeq)")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
