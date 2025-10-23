"""
Example: AAL Brain Network with ALN Models and Parameter Optimization

This script demonstrates how to:
1. Create a whole-brain ALN network using the AAL atlas (116 regions)
2. Load empirical functional connectivity data
3. Optimize network parameters to match the empirical FC
4. Analyze and visualize results

Usage:
    python example_aal_aln_optimization.py --fc_file your_fc_matrix.csv

Author: Generated with Claude Code
Date: 2025-10-23
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurolib.models.aal_aln_network import AALALNNetwork, create_example_fc_matrix
from neurolib.utils import functions as func


def plot_fc_comparison(empirical_fc, simulated_fc, save_path=None):
    """
    Plot comparison between empirical and simulated FC matrices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Empirical FC
    im1 = axes[0].imshow(empirical_fc, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Empirical FC')
    axes[0].set_xlabel('Brain Region')
    axes[0].set_ylabel('Brain Region')
    plt.colorbar(im1, ax=axes[0])

    # Simulated FC
    im2 = axes[1].imshow(simulated_fc, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Simulated FC')
    axes[1].set_xlabel('Brain Region')
    axes[1].set_ylabel('Brain Region')
    plt.colorbar(im2, ax=axes[1])

    # Scatter plot
    # Get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices_from(empirical_fc, k=1)
    emp_values = empirical_fc[triu_indices]
    sim_values = simulated_fc[triu_indices]

    axes[2].scatter(emp_values, sim_values, alpha=0.3, s=1)
    axes[2].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect fit')
    axes[2].set_xlabel('Empirical FC')
    axes[2].set_ylabel('Simulated FC')
    axes[2].set_title(f'FC Correlation\nr = {func.matrix_correlation(empirical_fc, simulated_fc):.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"FC comparison plot saved to {save_path}")

    return fig


def plot_firing_rates(firing_rates, region_names=None, save_path=None):
    """
    Plot firing rate time series for all regions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Heatmap of all firing rates
    im = axes[0].imshow(firing_rates, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Brain Region')
    axes[0].set_title('Firing Rates Across All Regions')
    plt.colorbar(im, ax=axes[0], label='Firing Rate (Hz)')

    # Sample firing rates for a few regions
    n_regions = firing_rates.shape[0]
    sample_indices = np.linspace(0, n_regions-1, min(5, n_regions), dtype=int)
    time = np.arange(firing_rates.shape[1])

    for idx in sample_indices:
        label = region_names[idx] if region_names else f'Region {idx}'
        axes[1].plot(time, firing_rates[idx, :], label=label, alpha=0.7)

    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Firing Rate (Hz)')
    axes[1].set_title('Sample Firing Rate Time Series')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Firing rates plot saved to {save_path}")

    return fig


def example_basic_simulation():
    """
    Example 1: Basic simulation without optimization.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Simulation")
    print("="*70)

    # Create example FC data
    logging.info("Creating example FC matrix...")
    empirical_fc = create_example_fc_matrix(n_regions=116, output_file="example_fc_116x116.csv")

    # Create random structural connectivity
    logging.info("Creating structural connectivity matrix...")
    np.random.seed(42)
    connectivity = np.random.rand(116, 116) * 0.3
    connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
    np.fill_diagonal(connectivity, 0)  # No self-connections

    # Create network
    logging.info("Creating AAL-ALN network...")
    network = AALALNNetwork(
        connectivity_matrix=connectivity,
        n_regions=116,
    )

    # Load empirical FC
    network.load_empirical_fc("example_fc_116x116.csv")

    # Run simulation
    logging.info("Running simulation...")
    results = network.run_simulation(duration=10000, dt=0.1)  # 10 seconds

    # Compute FC similarity
    fc_similarity = network.compute_fc_similarity(results['simulated_fc'])
    logging.info(f"FC similarity (correlation): {fc_similarity:.4f}")

    # Plot results
    plot_fc_comparison(
        network.empirical_fc,
        results['simulated_fc'],
        save_path='fc_comparison_basic.png'
    )

    plot_firing_rates(
        results['firing_rates_exc'],
        region_names=network.region_names,
        save_path='firing_rates_basic.png'
    )

    # Save results
    network.save_results('basic_simulation_results.npz', include_timeseries=True)

    logging.info("Basic simulation completed!")

    return network


def example_region_specific_parameters():
    """
    Example 2: Network with region-specific parameters.
    """
    print("\n" + "="*70)
    print("Example 2: Region-Specific Parameters")
    print("="*70)

    # Create different parameters for each region
    n_regions = 116

    # Create parameter variations
    # For example, frontal regions have higher excitability
    exc_params_list = []
    for i in range(n_regions):
        params = {
            'Ke': 800.0 + np.random.randn() * 50,  # Add some variability
            'Ki': 200.0 + np.random.randn() * 20,
            'c_gl': 0.4 + np.random.randn() * 0.05,
        }
        exc_params_list.append(params)

    # Create network with region-specific parameters
    connectivity = np.random.rand(n_regions, n_regions) * 0.3
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)

    network = AALALNNetwork(
        connectivity_matrix=connectivity,
        n_regions=n_regions,
        exc_params=exc_params_list,  # Region-specific parameters
    )

    # Get region information
    region_info = network.get_region_info()
    logging.info(f"\nRegion information:\n{region_info.head()}")

    logging.info("Region-specific network created successfully!")

    return network


def example_parameter_optimization(fc_file=None):
    """
    Example 3: Full parameter optimization to match empirical FC.
    """
    print("\n" + "="*70)
    print("Example 3: Parameter Optimization")
    print("="*70)

    # Use provided FC file or create example
    if fc_file is None or not os.path.exists(fc_file):
        logging.info("No FC file provided, creating example...")
        empirical_fc = create_example_fc_matrix(n_regions=116, output_file="example_fc_116x116.csv")
        fc_file = "example_fc_116x116.csv"
    else:
        empirical_fc = None

    # Create structural connectivity
    # In practice, this should come from DTI data
    n_regions = 116
    np.random.seed(42)
    connectivity = np.random.rand(n_regions, n_regions) * 0.3
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)

    # Create distance-based delays
    delay_matrix = np.random.rand(n_regions, n_regions) * 10  # 0-10 ms delays
    delay_matrix = (delay_matrix + delay_matrix.T) / 2

    # Create network
    logging.info("Creating AAL-ALN network...")
    network = AALALNNetwork(
        connectivity_matrix=connectivity,
        delay_matrix=delay_matrix,
        n_regions=n_regions,
    )

    # Load empirical FC
    network.load_empirical_fc(fc_file)

    # Define parameter space for optimization
    # You can optimize global parameters or region-specific parameters
    param_space = {
        'c_gl': (0.2, 0.6),          # Global coupling strength
        'Ke_gl': (100, 400),         # Global excitatory connections
        'sigmae_ext': (1.0, 2.5),    # External noise
        'tau_se': (1.0, 4.0),        # Excitatory time constant
        # Add region-specific parameters if needed:
        # 'node_0_Ke': (600, 1000),  # Region 0 specific
        # 'node_1_Ke': (600, 1000),  # Region 1 specific
    }

    # Run optimization
    logging.info("Starting parameter optimization...")
    logging.info("Note: This is a quick example. For real optimization, use larger population and more generations.")

    evolution = network.optimize_parameters(
        param_space=param_space,
        population_size=20,         # Use larger for real optimization (e.g., 100-200)
        n_generations=10,            # Use more for real optimization (e.g., 50-100)
        output_file="aal_aln_optimization.hdf",
        use_bold=False,              # Set to True for BOLD-based optimization (slower)
        simulation_duration=20000,   # 20 seconds (use longer for real optimization)
        verbose=True,
    )

    # Get best parameters
    best_individual = evolution.dfPop.iloc[evolution.dfPop.score_0.idxmax()]
    best_fitness = best_individual.score_0
    best_params = {name: best_individual[name] for name in param_space.keys()}

    logging.info(f"\nOptimization Results:")
    logging.info(f"Best fitness (FC correlation): {best_fitness:.4f}")
    logging.info(f"Best parameters:")
    for param_name, param_value in best_params.items():
        logging.info(f"  {param_name}: {param_value:.4f}")

    # Update network with best parameters
    network.update_parameters(best_params)

    # Run final simulation with optimized parameters
    logging.info("Running final simulation with optimized parameters...")
    results = network.run_simulation(duration=30000)  # 30 seconds

    # Plot results
    plot_fc_comparison(
        network.empirical_fc,
        results['simulated_fc'],
        save_path='fc_comparison_optimized.png'
    )

    plot_firing_rates(
        results['firing_rates_exc'],
        region_names=network.region_names,
        save_path='firing_rates_optimized.png'
    )

    # Plot optimization progress
    plot_optimization_progress(evolution, save_path='optimization_progress.png')

    # Save results
    network.save_results('optimized_simulation_results.npz', include_timeseries=True)

    logging.info("Parameter optimization completed!")

    return network, evolution


def plot_optimization_progress(evolution, save_path=None):
    """
    Plot the progress of evolutionary optimization.
    """
    gens, all_scores = evolution.getScoresDuringEvolution(reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Score over generations
    mean_scores = np.nanmean(all_scores, axis=1)
    max_scores = np.nanmax(all_scores, axis=1)
    min_scores = np.nanmin(all_scores, axis=1)

    axes[0].plot(gens, mean_scores, 'b-', linewidth=2, label='Mean')
    axes[0].plot(gens, max_scores, 'g-', linewidth=2, label='Best')
    axes[0].fill_between(gens, min_scores, max_scores, alpha=0.3)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness (FC Correlation)')
    axes[0].set_title('Optimization Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final population distribution
    final_scores = all_scores[-1, :]
    axes[1].hist(final_scores[~np.isnan(final_scores)], bins=20, edgecolor='black')
    axes[1].axvline(np.nanmax(final_scores), color='r', linestyle='--',
                    linewidth=2, label=f'Best: {np.nanmax(final_scores):.3f}')
    axes[1].set_xlabel('Fitness (FC Correlation)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Final Population Fitness Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Optimization progress plot saved to {save_path}")

    return fig


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='AAL Brain Network with ALN Models - Example Script'
    )
    parser.add_argument(
        '--fc_file',
        type=str,
        default=None,
        help='Path to empirical FC matrix CSV file (116x116)'
    )
    parser.add_argument(
        '--example',
        type=str,
        choices=['basic', 'region_specific', 'optimize', 'all'],
        default='all',
        help='Which example to run'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        if args.example in ['basic', 'all']:
            example_basic_simulation()

        if args.example in ['region_specific', 'all']:
            example_region_specific_parameters()

        if args.example in ['optimize', 'all']:
            example_parameter_optimization(args.fc_file)

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)

    except Exception as e:
        logging.error(f"Error running examples: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
