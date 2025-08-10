"""
Performance Benchmarking Utilities for Plinko Dynamics

Advanced benchmarking and profiling tools for quantum simulations.
"""

import time
import psutil
import tracemalloc
from typing import Dict, Any, Callable, Optional, List, Tuple
from functools import wraps
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    execution_time: float
    memory_peak: float
    memory_current: float
    cpu_percent: float
    timestamp: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "execution_time": self.execution_time,
            "memory_peak_mb": self.memory_peak / 1024 / 1024,
            "memory_current_mb": self.memory_current / 1024 / 1024,
            "cpu_percent": self.cpu_percent,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "error": self.error
        }


class PerformanceBenchmark:
    """Advanced performance benchmarking for quantum simulations."""
    
    def __init__(self, warmup_runs: int = 2):
        """
        Initialize benchmark suite.
        
        Args:
            warmup_runs: Number of warmup runs before actual benchmark
        """
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
        
    def benchmark_function(self, 
                          func: Callable,
                          *args,
                          name: Optional[str] = None,
                          runs: int = 10,
                          **kwargs) -> BenchmarkResult:
        """
        Benchmark a single function with memory and CPU tracking.
        
        Args:
            func: Function to benchmark
            *args: Positional arguments for function
            name: Name for the benchmark
            runs: Number of benchmark runs
            **kwargs: Keyword arguments for function
            
        Returns:
            BenchmarkResult with performance metrics
        """
        name = name or func.__name__
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Start tracking
        tracemalloc.start()
        process = psutil.Process()
        
        execution_times = []
        cpu_percentages = []
        error_msg = None
        
        try:
            for _ in range(runs):
                # CPU tracking
                cpu_before = process.cpu_percent()
                
                # Time tracking
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()
                
                cpu_after = process.cpu_percent()
                
                execution_times.append(end_time - start_time)
                cpu_percentages.append((cpu_before + cpu_after) / 2)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Benchmark failed for {name}: {e}")
        
        # Memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Create result
        result = BenchmarkResult(
            name=name,
            execution_time=np.mean(execution_times) if execution_times else 0,
            memory_peak=peak,
            memory_current=current,
            cpu_percent=np.mean(cpu_percentages) if cpu_percentages else 0,
            timestamp=datetime.now().isoformat(),
            parameters={"args": str(args), "kwargs": str(kwargs)},
            error=error_msg
        )
        
        self.results.append(result)
        return result
    
    def benchmark_quantum_circuits(self, qgb_instance) -> Dict[str, BenchmarkResult]:
        """
        Benchmark different quantum circuit types.
        
        Args:
            qgb_instance: QuantumGaltonBoard instance
            
        Returns:
            Dictionary of benchmark results
        """
        circuit_results = {}
        
        # Benchmark Gaussian circuit
        result = self.benchmark_function(
            qgb_instance.run_simulation,
            "gaussian",
            name="gaussian_circuit",
            runs=5
        )
        circuit_results["gaussian"] = result
        
        # Benchmark Exponential circuit
        result = self.benchmark_function(
            qgb_instance.run_simulation,
            "exponential",
            name="exponential_circuit",
            runs=5
        )
        circuit_results["exponential"] = result
        
        # Benchmark Hadamard walk
        result = self.benchmark_function(
            qgb_instance.run_simulation,
            "hadamard",
            name="hadamard_circuit",
            runs=5
        )
        circuit_results["hadamard"] = result
        
        return circuit_results
    
    def benchmark_scaling(self,
                         qgb_class,
                         layer_sizes: List[int] = [2, 3, 4, 5],
                         shots: int = 1000) -> Dict[int, BenchmarkResult]:
        """
        Benchmark scaling with different layer sizes.
        
        Args:
            qgb_class: QuantumGaltonBoard class
            layer_sizes: List of layer sizes to test
            shots: Number of shots per simulation
            
        Returns:
            Dictionary mapping layer size to benchmark results
        """
        scaling_results = {}
        
        for n_layers in layer_sizes:
            try:
                qgb = qgb_class(n_layers=n_layers, n_shots=shots)
                result = self.benchmark_function(
                    qgb.run_simulation,
                    "gaussian",
                    name=f"scaling_{n_layers}_layers",
                    runs=3
                )
                scaling_results[n_layers] = result
            except Exception as e:
                logger.error(f"Scaling benchmark failed for {n_layers} layers: {e}")
                scaling_results[n_layers] = BenchmarkResult(
                    name=f"scaling_{n_layers}_layers",
                    execution_time=0,
                    memory_peak=0,
                    memory_current=0,
                    cpu_percent=0,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
        
        return scaling_results
    
    def compare_implementations(self,
                               implementations: Dict[str, Callable],
                               *args,
                               **kwargs) -> Dict[str, BenchmarkResult]:
        """
        Compare different implementations of the same functionality.
        
        Args:
            implementations: Dictionary of implementation name to function
            *args: Arguments to pass to each implementation
            **kwargs: Keyword arguments to pass
            
        Returns:
            Dictionary of benchmark results
        """
        comparison_results = {}
        
        for name, impl in implementations.items():
            result = self.benchmark_function(
                impl,
                *args,
                name=name,
                runs=5,
                **kwargs
            )
            comparison_results[name] = result
        
        return comparison_results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Total Benchmarks: {len(self.results)}",
            "",
            "-" * 80,
            "BENCHMARK RESULTS",
            "-" * 80,
        ]
        
        for result in self.results:
            report_lines.extend([
                f"\nBenchmark: {result.name}",
                f"  Execution Time: {result.execution_time:.4f} seconds",
                f"  Memory Peak: {result.memory_peak / 1024 / 1024:.2f} MB",
                f"  Memory Current: {result.memory_current / 1024 / 1024:.2f} MB",
                f"  CPU Usage: {result.cpu_percent:.1f}%",
                f"  Timestamp: {result.timestamp}",
            ])
            
            if result.error:
                report_lines.append(f"  ERROR: {result.error}")
        
        # Statistical Summary
        if self.results:
            exec_times = [r.execution_time for r in self.results if not r.error]
            memory_peaks = [r.memory_peak for r in self.results if not r.error]
            
            if exec_times:
                report_lines.extend([
                    "",
                    "-" * 80,
                    "STATISTICAL SUMMARY",
                    "-" * 80,
                    f"Average Execution Time: {np.mean(exec_times):.4f} seconds",
                    f"Std Dev Execution Time: {np.std(exec_times):.4f} seconds",
                    f"Min Execution Time: {np.min(exec_times):.4f} seconds",
                    f"Max Execution Time: {np.max(exec_times):.4f} seconds",
                    "",
                    f"Average Memory Peak: {np.mean(memory_peaks) / 1024 / 1024:.2f} MB",
                    f"Max Memory Peak: {np.max(memory_peaks) / 1024 / 1024:.2f} MB",
                ])
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            
            # Also save JSON version
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        return report
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Create visualization of benchmark results.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                logger.warning("No results to plot")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Performance Benchmark Results", fontsize=16)
            
            # Execution times
            ax1 = axes[0, 0]
            names = [r.name for r in self.results]
            times = [r.execution_time for r in self.results]
            ax1.bar(range(len(names)), times, color='purple', alpha=0.7)
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels(names, rotation=45, ha='right')
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Execution Times')
            ax1.grid(True, alpha=0.3)
            
            # Memory usage
            ax2 = axes[0, 1]
            memory = [r.memory_peak / 1024 / 1024 for r in self.results]
            ax2.bar(range(len(names)), memory, color='navy', alpha=0.7)
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.set_ylabel('Memory Peak (MB)')
            ax2.set_title('Memory Usage')
            ax2.grid(True, alpha=0.3)
            
            # CPU usage
            ax3 = axes[1, 0]
            cpu = [r.cpu_percent for r in self.results]
            ax3.bar(range(len(names)), cpu, color='teal', alpha=0.7)
            ax3.set_xticks(range(len(names)))
            ax3.set_xticklabels(names, rotation=45, ha='right')
            ax3.set_ylabel('CPU Usage (%)')
            ax3.set_title('CPU Utilization')
            ax3.grid(True, alpha=0.3)
            
            # Performance efficiency (inverse of time * memory)
            ax4 = axes[1, 1]
            efficiency = [1 / (r.execution_time * r.memory_peak / 1024 / 1024 + 1e-6) 
                         for r in self.results]
            ax4.bar(range(len(names)), efficiency, color='green', alpha=0.7)
            ax4.set_xticks(range(len(names)))
            ax4.set_xticklabels(names, rotation=45, ha='right')
            ax4.set_ylabel('Efficiency Score')
            ax4.set_title('Performance Efficiency')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def timeit(func: Callable) -> Callable:
    """
    Decorator for simple timing of functions.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def profile_memory(func: Callable) -> Callable:
    """
    Decorator for memory profiling of functions.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with memory profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"{func.__name__} memory - Current: {current/1024/1024:.2f}MB, "
                   f"Peak: {peak/1024/1024:.2f}MB")
        return result
    return wrapper
