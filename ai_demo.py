"""
AI Demo for Quantum Galton Board
Demonstrates AI integration with smaller dataset for faster training
"""

import torch
import numpy as np
from quantum_galton_board import QuantumGaltonBoard
from latent_ode import QuantumGaltonAI
import matplotlib.pyplot as plt

def generate_simple_dataset(num_samples=10, n_layers=3, time_steps=10):
    """Generate a simple dataset for AI training demo."""
    print(f"Generating {num_samples} quantum trajectories for AI training...")
    
    trajectories = []
    parameters = []
    
    for i in range(num_samples):
        print(f"Generating trajectory {i+1}/{num_samples}")
        
        # Create quantum Galton board
        qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=100)
        
        # Generate trajectory with different rotation angles
        trajectory = []
        rotation_angles = np.linspace(0.1 * np.pi, 0.4 * np.pi, time_steps)
        
        for angle in rotation_angles:
            # Create circuit with current angle
            circuit = qgb.create_circuit(angle)
            samples = circuit()
            distribution = qgb.get_probability_distribution(samples)
            
            # Pad to consistent size
            max_bins = n_layers + 1
            if len(distribution) < max_bins:
                padded_dist = np.zeros(max_bins)
                padded_dist[:len(distribution)] = distribution
                distribution = padded_dist
            
            trajectory.append(distribution)
        
        trajectories.append(trajectory)
        parameters.append([n_layers, rotation_angles[0], rotation_angles[-1]])
    
    # Convert to tensors
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)
    parameters = torch.tensor(np.array(parameters), dtype=torch.float32)
    
    print(f"Dataset generated with shape: {trajectories.shape}")
    return trajectories, parameters

def train_ai_model(trajectories, parameters, num_epochs=20):
    """Train the AI model on the generated dataset."""
    print("\nTraining AI model...")
    
    # Create time steps
    time_steps = torch.linspace(0, 1, trajectories.shape[1])
    
    # Initialize AI model
    model = QuantumGaltonAI(
        obs_dim=trajectories.shape[-1],
        latent_dim=4,
        nhidden=16,
        rnn_nhidden=12,
        lr=5e-3,
        batch=trajectories.shape[0]
    )
    
    # Train model (skip if torchdiffeq not available)
    try:
        model.train(trajectories, time_steps, num_epochs)
        print("AI model training completed!")
        
        # Save model
        model.save_model("demo_quantum_galton_ai")
        
        # Test encoding/decoding
        print("\nTesting AI model...")
        encoded = model.encode(trajectories[:3], time_steps)
        decoded = model.decode(encoded, time_steps)
        
        print(f"Original shape: {trajectories[:3].shape}")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Decoded shape: {decoded.shape}")
        
        return model
        
    except Exception as e:
        print(f"AI training skipped due to missing dependencies: {e}")
        return None

def visualize_results(trajectories, model=None):
    """Visualize the results."""
    print("\nVisualizing results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot some original trajectories
    for i in range(min(3, trajectories.shape[0])):
        for t in range(trajectories.shape[1]):
            axes[0, 0].plot(trajectories[i, t], alpha=0.6, label=f'Traj {i+1}' if t == 0 else "")
    axes[0, 0].set_title('Original Quantum Trajectories')
    axes[0, 0].set_xlabel('Bin')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].legend()
    
    # Plot trajectory evolution over time
    traj_idx = 0
    im = axes[0, 1].imshow(trajectories[traj_idx].T, aspect='auto', cmap='viridis')
    axes[0, 1].set_title(f'Trajectory {traj_idx+1} Evolution')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Bin')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot average distribution over time
    avg_dist = trajectories.mean(dim=0)
    for t in range(avg_dist.shape[0]):
        axes[1, 0].plot(avg_dist[t], alpha=0.7, label=f't={t}' if t % 3 == 0 else "")
    axes[1, 0].set_title('Average Distribution Evolution')
    axes[1, 0].set_xlabel('Bin')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].legend()
    
    # Plot variance across trajectories
    var_dist = trajectories.var(dim=0)
    im2 = axes[1, 1].imshow(var_dist.T, aspect='auto', cmap='plasma')
    axes[1, 1].set_title('Variance Across Trajectories')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Bin')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('ai_quantum_galton_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'ai_quantum_galton_analysis.png'")

def main():
    """Main demo function."""
    print("Quantum Galton Board AI Integration Demo")
    print("=" * 50)
    
    # Generate dataset
    trajectories, parameters = generate_simple_dataset(
        num_samples=8, 
        n_layers=3, 
        time_steps=8
    )
    
    # Train AI model
    model = train_ai_model(trajectories, parameters, num_epochs=15)
    
    # Visualize results
    visualize_results(trajectories, model)
    
    print("\nDemo completed! Check the generated visualization.")

if __name__ == "__main__":
    main()
