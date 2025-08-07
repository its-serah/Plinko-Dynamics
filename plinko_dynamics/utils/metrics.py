"""
Comprehensive Distribution Metrics

Advanced statistical measures and comparison tools for analyzing
quantum and classical Galton board distributions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings


class DistributionMetrics:
    """
    Comprehensive metrics for analyzing and comparing probability distributions.
    
    This class provides a wide range of statistical measures and comparison
    tools for quantum and classical Galton board distributions.
    """
    
    @staticmethod
    def basic_statistics(distribution: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistical measures for a distribution.
        
        Args:
            distribution: Probability distribution array
            
        Returns:
            Dictionary containing basic statistics
        """
        if len(distribution) == 0:
            raise ValueError("Distribution cannot be empty")
        if not np.isclose(np.sum(distribution), 1.0, atol=0.1):
            warnings.warn(f"Distribution sums to {np.sum(distribution)}, expected ~1.0")
        
        x = np.arange(len(distribution))
        
        # Basic moments
        mean = np.sum(x * distribution)
        variance = np.sum((x - mean)**2 * distribution)
        std_dev = np.sqrt(variance)
        
        # Higher moments
        if std_dev > 0:
            skewness = np.sum(((x - mean) / std_dev)**3 * distribution)
            kurtosis = np.sum(((x - mean) / std_dev)**4 * distribution)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Additional measures
        mode_idx = np.argmax(distribution)
        mode = x[mode_idx]
        
        # Entropy
        eps = 1e-10
        entropy = -np.sum(distribution * np.log(distribution + eps))
        
        # Range and spread
        support = np.where(distribution > 0.001)[0]  # Non-negligible probability
        effective_range = support[-1] - support[0] + 1 if len(support) > 0 else 0
        
        return {
            'mean': float(mean),
            'median': DistributionMetrics._compute_median(distribution),
            'mode': float(mode),
            'variance': float(variance),
            'std_dev': float(std_dev),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'entropy': float(entropy),
            'min_prob': float(np.min(distribution)),
            'max_prob': float(np.max(distribution)),
            'effective_range': int(effective_range),
            'coefficient_of_variation': float(std_dev / mean) if mean > 0 else float('inf')
        }
    
    @staticmethod
    def _compute_median(distribution: np.ndarray) -> float:
        """Compute the median of a discrete distribution."""
        cumsum = np.cumsum(distribution)
        median_idx = np.searchsorted(cumsum, 0.5)
        return float(median_idx)
    
    @staticmethod
    def compare_distributions(dist1: np.ndarray, dist2: np.ndarray, 
                            labels: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Comprehensive comparison between two probability distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            labels: Optional labels for the distributions
            
        Returns:
            Dictionary containing comparison metrics
        """
        if len(dist1) != len(dist2):
            raise ValueError("Distributions must have the same length")
        
        labels = labels or ("Distribution 1", "Distribution 2")
        
        # Basic statistics for both distributions
        stats1 = DistributionMetrics.basic_statistics(dist1)
        stats2 = DistributionMetrics.basic_statistics(dist2)
        
        # Distance measures
        distances = DistributionMetrics.distance_measures(dist1, dist2)
        
        # Statistical tests
        statistical_tests = DistributionMetrics.statistical_tests(dist1, dist2)
        
        # Overlap measures
        overlap = DistributionMetrics.overlap_measures(dist1, dist2)
        
        return {
            'labels': labels,
            'statistics': {
                labels[0]: stats1,
                labels[1]: stats2
            },
            'distances': distances,
            'statistical_tests': statistical_tests,
            'overlap': overlap,
            'summary': {
                'mean_difference': abs(stats1['mean'] - stats2['mean']),
                'variance_ratio': stats1['variance'] / (stats2['variance'] + 1e-10),
                'most_similar_metric': min(distances, key=distances.get),
                'least_similar_metric': max(distances, key=distances.get)
            }
        }
    
    @staticmethod
    def distance_measures(dist1: np.ndarray, dist2: np.ndarray) -> Dict[str, float]:
        """
        Calculate various distance measures between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Dictionary of distance measures
        """
        eps = 1e-10
        
        # L1 and L2 distances
        l1_distance = np.sum(np.abs(dist1 - dist2))
        l2_distance = np.sqrt(np.sum((dist1 - dist2)**2))
        
        # Total Variation Distance
        tv_distance = 0.5 * l1_distance
        
        # Kullback-Leibler Divergences
        kl_1_to_2 = np.sum(dist1 * np.log((dist1 + eps) / (dist2 + eps)))
        kl_2_to_1 = np.sum(dist2 * np.log((dist2 + eps) / (dist1 + eps)))
        kl_symmetric = 0.5 * (kl_1_to_2 + kl_2_to_1)  # Jensen-Shannon divergence base
        
        # Jensen-Shannon Divergence
        m = 0.5 * (dist1 + dist2)
        js_divergence = 0.5 * np.sum(dist1 * np.log((dist1 + eps) / (m + eps))) + \
                       0.5 * np.sum(dist2 * np.log((dist2 + eps) / (m + eps)))
        
        # Hellinger Distance
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(dist1) - np.sqrt(dist2))**2))
        
        # Bhattacharyya Distance
        bc_coefficient = np.sum(np.sqrt(dist1 * dist2))
        bhattacharyya = -np.log(bc_coefficient + eps)
        
        # Chi-squared Distance
        chi_squared = np.sum((dist1 - dist2)**2 / (dist2 + eps))
        
        # Wasserstein Distance (Earth Mover's Distance)
        wasserstein = DistributionMetrics._wasserstein_distance(dist1, dist2)
        
        # Cosine Distance
        dot_product = np.dot(dist1, dist2)
        norm1 = np.linalg.norm(dist1)
        norm2 = np.linalg.norm(dist2)
        cosine_similarity = dot_product / (norm1 * norm2 + eps)
        cosine_distance = 1 - cosine_similarity
        
        return {
            'l1_distance': float(l1_distance),
            'l2_distance': float(l2_distance),
            'total_variation': float(tv_distance),
            'kl_divergence_1_to_2': float(kl_1_to_2),
            'kl_divergence_2_to_1': float(kl_2_to_1),
            'kl_symmetric': float(kl_symmetric),
            'jensen_shannon': float(js_divergence),
            'hellinger': float(hellinger),
            'bhattacharyya': float(bhattacharyya),
            'chi_squared': float(chi_squared),
            'wasserstein': float(wasserstein),
            'cosine_distance': float(cosine_distance)
        }
    
    @staticmethod
    def _wasserstein_distance(dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calculate 1-Wasserstein distance between two discrete distributions."""
        cumsum1 = np.cumsum(dist1)
        cumsum2 = np.cumsum(dist2)
        return float(np.sum(np.abs(cumsum1 - cumsum2)))
    
    @staticmethod
    def statistical_tests(dist1: np.ndarray, dist2: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical tests between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Dictionary containing test results
        """
        # Convert distributions to sample data for testing
        # (This is an approximation since we have probability distributions)
        n_samples = 1000
        
        try:
            # Generate samples from distributions
            x = np.arange(len(dist1))
            samples1 = np.random.choice(x, size=n_samples, p=dist1)
            samples2 = np.random.choice(x, size=n_samples, p=dist2)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(samples1, samples2)
            
            # Mann-Whitney U test
            mw_stat, mw_pvalue = stats.mannwhitneyu(samples1, samples2, alternative='two-sided')
            
            # Anderson-Darling test (if available)
            try:
                ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([samples1, samples2])
                ad_result = {
                    'statistic': float(ad_stat),
                    'critical_values': ad_critical.tolist(),
                    'significance_levels': ad_significance.tolist()
                }
            except:
                ad_result = None
            
            return {
                'kolmogorov_smirnov': {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_pvalue),
                    'significant_at_0.05': ks_pvalue < 0.05
                },
                'mann_whitney_u': {
                    'statistic': float(mw_stat),
                    'p_value': float(mw_pvalue),
                    'significant_at_0.05': mw_pvalue < 0.05
                },
                'anderson_darling': ad_result
            }
            
        except Exception as e:
            warnings.warn(f"Statistical tests failed: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def overlap_measures(dist1: np.ndarray, dist2: np.ndarray) -> Dict[str, float]:
        """
        Calculate overlap measures between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Dictionary of overlap measures
        """
        # Intersection (common area under both curves)
        intersection = np.sum(np.minimum(dist1, dist2))
        
        # Union
        union = np.sum(np.maximum(dist1, dist2))
        
        # Jaccard similarity
        jaccard = intersection / union if union > 0 else 0
        
        # Overlap coefficient
        overlap_coeff = intersection / min(np.sum(dist1), np.sum(dist2)) if min(np.sum(dist1), np.sum(dist2)) > 0 else 0
        
        # Bhattacharyya coefficient
        bhattacharyya_coeff = np.sum(np.sqrt(dist1 * dist2))
        
        return {
            'intersection': float(intersection),
            'union': float(union),
            'jaccard_similarity': float(jaccard),
            'overlap_coefficient': float(overlap_coeff),
            'bhattacharyya_coefficient': float(bhattacharyya_coeff)
        }
    
    @staticmethod
    def quantum_classical_analysis(quantum_dist: np.ndarray, 
                                 classical_dist: np.ndarray,
                                 theoretical_dist: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Specialized analysis for quantum vs classical distributions.
        
        Args:
            quantum_dist: Quantum Galton board distribution
            classical_dist: Classical Galton board distribution  
            theoretical_dist: Theoretical binomial distribution (optional)
            
        Returns:
            Comprehensive analysis results
        """
        # Basic comparison
        comparison = DistributionMetrics.compare_distributions(
            quantum_dist, classical_dist, ("Quantum", "Classical")
        )
        
        # Quantum advantage metrics
        quantum_advantage = {
            'spreads_comparison': {
                'quantum_std': comparison['statistics']['Quantum']['std_dev'],
                'classical_std': comparison['statistics']['Classical']['std_dev'],
                'ratio': comparison['statistics']['Quantum']['std_dev'] / 
                        (comparison['statistics']['Classical']['std_dev'] + 1e-10)
            },
            'entropy_comparison': {
                'quantum_entropy': comparison['statistics']['Quantum']['entropy'],
                'classical_entropy': comparison['statistics']['Classical']['entropy'],
                'difference': comparison['statistics']['Quantum']['entropy'] - 
                            comparison['statistics']['Classical']['entropy']
            }
        }
        
        result = {
            'basic_comparison': comparison,
            'quantum_advantage': quantum_advantage
        }
        
        # Add theoretical comparison if provided
        if theoretical_dist is not None:
            theoretical_comparison = {
                'quantum_vs_theoretical': DistributionMetrics.compare_distributions(
                    quantum_dist, theoretical_dist, ("Quantum", "Theoretical")
                ),
                'classical_vs_theoretical': DistributionMetrics.compare_distributions(
                    classical_dist, theoretical_dist, ("Classical", "Theoretical")
                )
            }
            result['theoretical_comparison'] = theoretical_comparison
        
        return result
    
    @staticmethod
    def circuit_performance_metrics(distributions: Dict[str, np.ndarray],
                                  target_dist: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze performance of different quantum circuits.
        
        Args:
            distributions: Dictionary of circuit results
            target_dist: Target distribution for comparison
            
        Returns:
            Performance analysis results
        """
        if not distributions:
            raise ValueError("No distributions provided")
        
        # Compare all circuits pairwise
        circuit_names = list(distributions.keys())
        pairwise_comparisons = {}
        
        for i, name1 in enumerate(circuit_names):
            for j, name2 in enumerate(circuit_names[i+1:], i+1):
                key = f"{name1}_vs_{name2}"
                pairwise_comparisons[key] = DistributionMetrics.compare_distributions(
                    distributions[name1], distributions[name2], (name1, name2)
                )
        
        # Statistics for each circuit
        circuit_stats = {}
        for name, dist in distributions.items():
            circuit_stats[name] = DistributionMetrics.basic_statistics(dist)
        
        result = {
            'circuit_statistics': circuit_stats,
            'pairwise_comparisons': pairwise_comparisons,
            'ranking': DistributionMetrics._rank_circuits_by_metrics(circuit_stats)
        }
        
        # Compare to target if provided
        if target_dist is not None:
            target_comparisons = {}
            for name, dist in distributions.items():
                target_comparisons[name] = DistributionMetrics.compare_distributions(
                    dist, target_dist, (name, "Target")
                )
            result['target_comparisons'] = target_comparisons
        
        return result
    
    @staticmethod
    def _rank_circuits_by_metrics(circuit_stats: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Rank circuits by various metrics."""
        rankings = {}
        
        for metric in ['entropy', 'std_dev', 'skewness', 'kurtosis']:
            if metric in next(iter(circuit_stats.values())):
                sorted_circuits = sorted(circuit_stats.items(), 
                                       key=lambda x: abs(x[1][metric]), reverse=True)
                rankings[f'highest_{metric}'] = [name for name, _ in sorted_circuits]
        
        return rankings
    
    @staticmethod
    def time_series_analysis(trajectory_data: np.ndarray, 
                           time_steps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze temporal evolution of distributions.
        
        Args:
            trajectory_data: Array of shape (n_trajectories, n_timesteps, n_bins)
            time_steps: Time step array (optional)
            
        Returns:
            Time series analysis results
        """
        if trajectory_data.ndim != 3:
            raise ValueError(f"Expected 3D trajectory data, got {trajectory_data.ndim}D")
        
        n_trajectories, n_timesteps, n_bins = trajectory_data.shape
        
        if time_steps is None:
            time_steps = np.arange(n_timesteps)
        
        # Average evolution
        mean_trajectory = np.mean(trajectory_data, axis=0)
        
        # Variance evolution
        var_trajectory = np.var(trajectory_data, axis=0)
        
        # Statistics evolution over time
        stats_evolution = []
        for t in range(n_timesteps):
            avg_dist = mean_trajectory[t]
            stats = DistributionMetrics.basic_statistics(avg_dist)
            stats['time'] = time_steps[t]
            stats_evolution.append(stats)
        
        # Convergence analysis
        final_dist = mean_trajectory[-1]
        convergence_metrics = []
        for t in range(1, n_timesteps):
            current_dist = mean_trajectory[t]
            distance = DistributionMetrics.distance_measures(current_dist, final_dist)
            convergence_metrics.append({
                'time': time_steps[t],
                'distance_to_final': distance['total_variation']
            })
        
        return {
            'mean_trajectory': mean_trajectory,
            'variance_trajectory': var_trajectory,
            'statistics_evolution': stats_evolution,
            'convergence_metrics': convergence_metrics,
            'stability_measure': np.mean([cm['distance_to_final'] for cm in convergence_metrics[-5:]]),
            'final_statistics': DistributionMetrics.basic_statistics(final_dist)
        }
