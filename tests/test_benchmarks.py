"""
Performance benchmark tests for Plinko Dynamics.

Run with: pytest tests/test_benchmarks.py --benchmark-only
"""

import pytest
import numpy as np
from plinko_dynamics.core.quantum_galton import QuantumGaltonBoard
from plinko_dynamics.ai.models import QuantumGaltonAI
from plinko_dynamics.visualization.plotter import PlinkoDynamicsVisualizer
from plinko_dynamics.utils.benchmarks import PerformanceBenchmark


class TestQuantumCircuitBenchmarks:
    """Benchmark tests for quantum circuits."""
    
    @pytest.fixture
    def small_qgb(self):
        """Small quantum Galton board for fast tests."""
        return QuantumGaltonBoard(n_layers=2, n_shots=100)
    
    @pytest.fixture
    def medium_qgb(self):
        """Medium quantum Galton board."""
        return QuantumGaltonBoard(n_layers=3, n_shots=500)
    
    @pytest.mark.benchmark(group="circuits")
    def test_gaussian_circuit_performance(self, benchmark, small_qgb):
        """Benchmark Gaussian circuit execution."""
        result = benchmark(small_qgb.run_simulation, "gaussian")
        assert len(result) == 100
    
    @pytest.mark.benchmark(group="circuits")
    def test_exponential_circuit_performance(self, benchmark, small_qgb):
        """Benchmark exponential circuit execution."""
        result = benchmark(small_qgb.run_simulation, "exponential")
        assert len(result) == 100
    
    @pytest.mark.benchmark(group="circuits")
    def test_hadamard_circuit_performance(self, benchmark, small_qgb):
        """Benchmark Hadamard walk circuit execution."""
        result = benchmark(small_qgb.run_simulation, "hadamard")
        assert len(result) == 100
    
    @pytest.mark.benchmark(group="scaling")
    @pytest.mark.parametrize("n_layers", [2, 3, 4])
    def test_circuit_scaling(self, benchmark, n_layers):
        """Benchmark circuit scaling with different layer counts."""
        qgb = QuantumGaltonBoard(n_layers=n_layers, n_shots=100)
        result = benchmark(qgb.run_simulation, "gaussian")
        assert len(result) == 100
    
    @pytest.mark.benchmark(group="shots")
    @pytest.mark.parametrize("n_shots", [100, 500, 1000])
    def test_shots_scaling(self, benchmark, n_shots):
        """Benchmark scaling with different shot counts."""
        qgb = QuantumGaltonBoard(n_layers=2, n_shots=n_shots)
        result = benchmark(qgb.run_simulation, "gaussian")
        assert len(result) == n_shots


class TestAIModelBenchmarks:
    """Benchmark tests for AI models."""
    
    @pytest.fixture
    def sample_trajectories(self):
        """Generate sample trajectory data."""
        n_trajectories = 10
        n_timesteps = 20
        n_bins = 5
        return np.random.rand(n_trajectories, n_timesteps, n_bins)
    
    @pytest.fixture
    def ai_model(self):
        """Create AI model instance."""
        return QuantumGaltonAI(obs_dim=5, latent_dim=4, nhidden=32)
    
    @pytest.mark.benchmark(group="ai")
    def test_ai_forward_pass(self, benchmark, ai_model, sample_trajectories):
        """Benchmark AI model forward pass."""
        def forward():
            return ai_model.encode_trajectories(sample_trajectories)
        
        result = benchmark(forward)
        assert result is not None
    
    @pytest.mark.benchmark(group="ai")
    def test_ai_training_step(self, benchmark, ai_model, sample_trajectories):
        """Benchmark single training step."""
        time_steps = np.linspace(0, 1, 20)
        
        def train_step():
            return ai_model.train_step(sample_trajectories, time_steps)
        
        result = benchmark(train_step)
        assert result is not None


class TestVisualizationBenchmarks:
    """Benchmark tests for visualization."""
    
    @pytest.fixture
    def sample_distributions(self):
        """Generate sample distribution data."""
        return {
            "gaussian": np.random.normal(0.5, 0.1, 5),
            "exponential": np.random.exponential(0.2, 5),
            "uniform": np.ones(5) / 5
        }
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return PlinkoDynamicsVisualizer(theme='purple_navy')
    
    @pytest.mark.benchmark(group="visualization")
    def test_distribution_plot_performance(self, benchmark, visualizer, sample_distributions):
        """Benchmark distribution plotting."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        def plot():
            return visualizer.plot_distribution_comparison(
                sample_distributions,
                title="Benchmark Test"
            )
        
        result = benchmark(plot)
        assert result is not None
    
    @pytest.mark.benchmark(group="visualization")
    def test_trajectory_plot_performance(self, benchmark, visualizer):
        """Benchmark trajectory plotting."""
        import matplotlib
        matplotlib.use('Agg')
        
        trajectories = np.random.rand(5, 10, 5)  # 5 trajectories, 10 timesteps, 5 bins
        
        def plot():
            return visualizer.plot_trajectory_evolution(
                trajectories,
                title="Trajectory Benchmark"
            )
        
        result = benchmark(plot)
        assert result is not None


class TestIntegratedBenchmarks:
    """Integration benchmark tests."""
    
    @pytest.mark.benchmark(group="integration")
    def test_full_pipeline(self, benchmark):
        """Benchmark full simulation pipeline."""
        def run_pipeline():
            # Create quantum board
            qgb = QuantumGaltonBoard(n_layers=2, n_shots=100)
            
            # Run simulations
            gaussian_samples = qgb.run_simulation("gaussian")
            exponential_samples = qgb.run_simulation("exponential")
            
            # Get distributions
            gaussian_dist = qgb.get_probability_distribution(gaussian_samples)
            exponential_dist = qgb.get_probability_distribution(exponential_samples)
            
            # Create visualization
            import matplotlib
            matplotlib.use('Agg')
            visualizer = PlinkoDynamicsVisualizer()
            fig = visualizer.plot_distribution_comparison({
                "gaussian": gaussian_dist,
                "exponential": exponential_dist
            })
            
            return gaussian_dist, exponential_dist
        
        result = benchmark(run_pipeline)
        assert result is not None
        assert len(result) == 2
    
    @pytest.mark.benchmark(group="memory")
    def test_memory_usage(self, benchmark):
        """Benchmark memory usage for large simulation."""
        def large_simulation():
            qgb = QuantumGaltonBoard(n_layers=4, n_shots=5000)
            samples = qgb.run_simulation("gaussian")
            distribution = qgb.get_probability_distribution(samples)
            return distribution
        
        result = benchmark(large_simulation)
        assert result is not None
        assert np.isclose(np.sum(result), 1.0, atol=0.01)


class TestCustomBenchmarks:
    """Custom benchmarking using PerformanceBenchmark class."""
    
    def test_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite."""
        benchmark = PerformanceBenchmark(warmup_runs=1)
        
        # Test quantum circuits
        qgb = QuantumGaltonBoard(n_layers=3, n_shots=1000)
        circuit_results = benchmark.benchmark_quantum_circuits(qgb)
        
        assert "gaussian" in circuit_results
        assert "exponential" in circuit_results
        assert "hadamard" in circuit_results
        
        # Test scaling
        scaling_results = benchmark.benchmark_scaling(
            QuantumGaltonBoard,
            layer_sizes=[2, 3],
            shots=100
        )
        
        assert 2 in scaling_results
        assert 3 in scaling_results
        
        # Generate report
        report = benchmark.generate_report()
        assert "PERFORMANCE BENCHMARK REPORT" in report
        assert len(benchmark.results) > 0
    
    def test_comparison_benchmark(self):
        """Test comparing different implementations."""
        benchmark = PerformanceBenchmark()
        
        def impl1():
            return np.random.rand(100)
        
        def impl2():
            return np.ones(100) * 0.5
        
        def impl3():
            return np.zeros(100)
        
        implementations = {
            "random": impl1,
            "constant": impl2,
            "zeros": impl3
        }
        
        results = benchmark.compare_implementations(implementations)
        
        assert "random" in results
        assert "constant" in results
        assert "zeros" in results
        
        # Constant and zeros should be faster than random
        assert results["zeros"].execution_time <= results["random"].execution_time


if __name__ == "__main__":
    # Run benchmarks with detailed output
    pytest.main([
        __file__,
        "--benchmark-only",
        "--benchmark-verbose",
        "--benchmark-sort=time",
        "--benchmark-save=benchmark",
        "--benchmark-save-data",
        "-v"
    ])
