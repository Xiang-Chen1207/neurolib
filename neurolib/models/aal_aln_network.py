"""
AAL Brain Template with ALN Neural Mass Models

This module implements a brain network using the AAL (Automated Anatomical Labeling) atlas
where each brain region is represented by an ALN (Adaptive Linear-Nonlinear) neural mass model.
The network supports:
- Individual parameter settings for each brain region
- Custom structural connectivity matrices
- Parameter optimization to match empirical functional connectivity

Author: Generated with Claude Code
Date: 2025-10-23
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

from ..models.multimodel.builder.aln import ALNNetwork, ALN_EXC_DEFAULT_PARAMS, ALN_INH_DEFAULT_PARAMS
from ..utils.atlases import AutomatedAnatomicalParcellation2
from ..optimize.evolution import Evolution
from ..utils.parameterSpace import ParameterSpace
from ..utils import functions as func


class AALALNNetwork:
    """
    Whole-brain ALN network using AAL brain template.

    This class creates a brain network where each region in the AAL atlas
    is represented by one ALN neural mass model (with excitatory and inhibitory populations).

    Attributes:
        n_regions (int): Number of brain regions (default: 116 for AAL2 without cerebellum)
        atlas (AutomatedAnatomicalParcellation2): AAL2 atlas object
        network (ALNNetwork): The ALN network model
        empirical_fc (np.ndarray): Empirical functional connectivity matrix
        region_names (List[str]): Names of brain regions
    """

    def __init__(
        self,
        connectivity_matrix: Optional[np.ndarray] = None,
        delay_matrix: Optional[np.ndarray] = None,
        n_regions: int = 116,
        exc_params: Optional[Union[Dict, List[Dict]]] = None,
        inh_params: Optional[Union[Dict, List[Dict]]] = None,
        use_subcortical: bool = True,
    ):
        """
        Initialize AAL-based ALN network.

        Parameters:
            connectivity_matrix: Structural connectivity matrix (n_regions x n_regions).
                                If None, uses identity matrix (no inter-regional connections).
            delay_matrix: Delay matrix in ms (n_regions x n_regions).
                         If None, uses zeros (instantaneous transmission).
            n_regions: Number of brain regions to use (default: 116 for AAL2).
            exc_params: Parameters for excitatory populations. Can be:
                       - None: use default parameters for all regions
                       - Dict: use same parameters for all regions
                       - List[Dict]: region-specific parameters (length must equal n_regions)
            inh_params: Parameters for inhibitory populations (same format as exc_params).
            use_subcortical: Whether to include subcortical regions (default: True).
                            If False, uses only cortical regions.
        """
        self.n_regions = n_regions
        self.use_subcortical = use_subcortical

        # Initialize AAL2 atlas
        logging.info("Initializing AAL2 atlas...")
        self.atlas = AutomatedAnatomicalParcellation2()

        # Select regions based on user preference
        if use_subcortical:
            # Use first n_regions (usually 116: all regions except cerebellum)
            self.region_indices = list(range(min(n_regions, len(self.atlas))))
        else:
            # Use only cortical regions
            self.region_indices = self.atlas.cortex[:n_regions]

        self.n_regions = len(self.region_indices)
        self.region_names = [self.atlas[i] for i in self.region_indices]

        logging.info(f"Using {self.n_regions} brain regions from AAL2 atlas")

        # Get region coordinates (centers of brain regions)
        self.region_coords = self.atlas.coords()
        if self.region_coords is not None:
            self.region_coords = [self.region_coords[i] for i in range(self.n_regions)]

        # Initialize connectivity matrices
        if connectivity_matrix is None:
            logging.warning("No connectivity matrix provided. Using identity matrix (no connections).")
            connectivity_matrix = np.eye(self.n_regions) * 0.1  # Small self-connections
        else:
            assert connectivity_matrix.shape == (self.n_regions, self.n_regions), \
                f"Connectivity matrix shape {connectivity_matrix.shape} doesn't match n_regions={self.n_regions}"

        if delay_matrix is None:
            logging.info("No delay matrix provided. Using zero delays.")
            delay_matrix = np.zeros((self.n_regions, self.n_regions))
        else:
            assert delay_matrix.shape == (self.n_regions, self.n_regions), \
                f"Delay matrix shape {delay_matrix.shape} doesn't match n_regions={self.n_regions}"

        self.connectivity_matrix = connectivity_matrix
        self.delay_matrix = delay_matrix

        # Prepare per-region parameters
        self.exc_params = self._prepare_region_params(exc_params, ALN_EXC_DEFAULT_PARAMS)
        self.inh_params = self._prepare_region_params(inh_params, ALN_INH_DEFAULT_PARAMS)

        # Create ALN network
        logging.info("Creating ALN network...")
        self.network = ALNNetwork(
            connectivity_matrix=self.connectivity_matrix,
            delay_matrix=self.delay_matrix,
            exc_mass_params=self.exc_params,
            inh_mass_params=self.inh_params,
        )

        # Storage for empirical data
        self.empirical_fc = None

        logging.info("AAL-ALN network initialized successfully!")

    def _prepare_region_params(
        self,
        params: Optional[Union[Dict, List[Dict]]],
        default_params: Dict
    ) -> List[Dict]:
        """
        Prepare per-region parameter list.

        Parameters:
            params: User-provided parameters
            default_params: Default parameter dictionary

        Returns:
            List of parameter dictionaries (one per region)
        """
        if params is None:
            # Use default for all regions
            return [default_params.copy() for _ in range(self.n_regions)]
        elif isinstance(params, dict):
            # Use same custom params for all regions
            return [params.copy() for _ in range(self.n_regions)]
        elif isinstance(params, list):
            # Use region-specific params
            assert len(params) == self.n_regions, \
                f"Parameter list length {len(params)} doesn't match n_regions={self.n_regions}"
            return params
        else:
            raise ValueError(f"Invalid params type: {type(params)}")

    def load_empirical_fc(self, fc_file: str, file_format: str = 'csv') -> np.ndarray:
        """
        Load empirical functional connectivity matrix from file.

        Parameters:
            fc_file: Path to FC matrix file
            file_format: File format ('csv', 'npy', or 'mat')

        Returns:
            Functional connectivity matrix (n_regions x n_regions)
        """
        logging.info(f"Loading empirical FC from {fc_file}...")

        if file_format == 'csv':
            fc_matrix = pd.read_csv(fc_file, header=None).values
        elif file_format == 'npy':
            fc_matrix = np.load(fc_file)
        elif file_format == 'mat':
            import scipy.io
            mat_data = scipy.io.loadmat(fc_file)
            # Try common key names
            for key in ['fc', 'FC', 'correlation', 'corr']:
                if key in mat_data:
                    fc_matrix = mat_data[key]
                    break
            else:
                raise ValueError(f"Could not find FC matrix in .mat file. Available keys: {mat_data.keys()}")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Validate shape
        assert fc_matrix.shape == (self.n_regions, self.n_regions), \
            f"FC matrix shape {fc_matrix.shape} doesn't match n_regions={self.n_regions}"

        # Handle NaN values
        fc_matrix = np.nan_to_num(fc_matrix)

        # Ensure symmetric (use average of upper and lower triangle)
        fc_matrix = (fc_matrix + fc_matrix.T) / 2

        # Set diagonal to 1
        np.fill_diagonal(fc_matrix, 1.0)

        self.empirical_fc = fc_matrix
        logging.info(f"Empirical FC loaded successfully. Shape: {fc_matrix.shape}")

        return fc_matrix

    def run_simulation(
        self,
        duration: float = 60000,  # ms (60 seconds)
        dt: float = 0.1,
        chunkwise: bool = False,
        bold: bool = False,
    ) -> Dict:
        """
        Run network simulation.

        Parameters:
            duration: Simulation duration in milliseconds
            dt: Integration time step in milliseconds
            chunkwise: Whether to run simulation in chunks (for long simulations)
            bold: Whether to compute BOLD signal

        Returns:
            Dictionary with simulation outputs
        """
        logging.info(f"Running simulation for {duration}ms...")

        self.network.params['duration'] = duration
        self.network.params['dt'] = dt

        if chunkwise:
            self.network.run(chunkwise=True, bold=bold)
        else:
            self.network.run(bold=bold)

        # Compute functional connectivity from simulated data
        if bold and hasattr(self.network, 'BOLD'):
            # Use BOLD signal for FC
            simulated_fc = func.fc(self.network.BOLD.BOLD)
        else:
            # Use firing rates for FC
            simulated_fc = func.fc(self.network.outputs.r_mean_EXC)

        return {
            'network': self.network,
            'simulated_fc': simulated_fc,
            'firing_rates_exc': self.network.outputs.r_mean_EXC,
            'firing_rates_inh': self.network.outputs.r_mean_INH,
        }

    def compute_fc_similarity(self, simulated_fc: np.ndarray) -> float:
        """
        Compute similarity between simulated and empirical FC.

        Parameters:
            simulated_fc: Simulated functional connectivity matrix

        Returns:
            Pearson correlation coefficient
        """
        if self.empirical_fc is None:
            raise ValueError("No empirical FC loaded. Use load_empirical_fc() first.")

        return func.matrix_correlation(simulated_fc, self.empirical_fc)

    def optimize_parameters(
        self,
        param_space: Dict[str, Tuple[float, float]],
        population_size: int = 100,
        n_generations: int = 50,
        initial_population_size: Optional[int] = None,
        output_file: str = "aal_aln_optimization.hdf",
        use_bold: bool = False,
        simulation_duration: float = 60000,  # ms
        verbose: bool = True,
    ) -> Evolution:
        """
        Optimize network parameters to match empirical functional connectivity.

        Parameters:
            param_space: Dictionary defining parameter search space.
                        Keys are parameter names (with node index suffix if region-specific).
                        Values are (min, max) tuples.
                        Example: {
                            'c_gl': (0.1, 0.6),  # Global coupling (same for all)
                            'node_0_Ke': (600, 1000),  # Region-specific Ke for node 0
                            'node_1_Ke': (600, 1000),  # Region-specific Ke for node 1
                        }
            population_size: Number of individuals per generation
            n_generations: Number of generations to evolve
            initial_population_size: Size of initial population (default: 4*population_size)
            output_file: HDF5 file to store optimization results
            use_bold: Whether to use BOLD signal (slower but more realistic)
            simulation_duration: Duration of each simulation in ms
            verbose: Print progress information

        Returns:
            Evolution object with optimization results
        """
        if self.empirical_fc is None:
            raise ValueError("No empirical FC loaded. Use load_empirical_fc() first.")

        if initial_population_size is None:
            initial_population_size = population_size * 4

        logging.info(f"Starting parameter optimization...")
        logging.info(f"Parameter space: {param_space}")
        logging.info(f"Population size: {population_size}, Generations: {n_generations}")

        # Create ParameterSpace object
        param_names = list(param_space.keys())
        param_ranges = [param_space[name] for name in param_names]
        pspace = ParameterSpace(param_names, param_ranges)

        # Define evaluation function
        def evaluate_simulation(traj):
            """Evaluate fitness of a parameter set."""
            # Get model with updated parameters
            model = evolution.getModelFromTraj(traj)

            # Quick pre-check: simulate briefly to check if activity is reasonable
            model.params['duration'] = 3000  # 3 seconds
            model.run()

            # Check if firing rates are in reasonable range
            max_rate = np.max(model.outputs.r_mean_EXC[:, model.t > 500])
            if max_rate > 160 or max_rate < 0.1:
                # Activity too high or too low, return poor fitness
                return (0.0,), {}

            # Full simulation
            model.params['duration'] = simulation_duration
            if use_bold:
                model.run(chunkwise=True, bold=True)
                simulated_fc = func.fc(model.BOLD.BOLD[:, 5:])  # Skip initial transient
            else:
                model.run()
                simulated_fc = func.fc(model.outputs.r_mean_EXC)

            # Compute fitness as correlation with empirical FC
            fitness = func.matrix_correlation(simulated_fc, self.empirical_fc)

            # Handle NaN fitness
            if np.isnan(fitness):
                fitness = 0.0

            return (fitness,), {}

        # Create Evolution object
        evolution = Evolution(
            evaluate_simulation,
            pspace,
            algorithm='nsga2',
            weightList=[1.0],  # Maximize FC correlation
            model=self.network,
            POP_INIT_SIZE=initial_population_size,
            POP_SIZE=population_size,
            NGEN=n_generations,
            filename=output_file,
        )

        # Run optimization
        logging.info("Running evolutionary optimization...")
        evolution.run(verbose=verbose)

        # Get best parameters
        best_individual = evolution.dfPop.iloc[evolution.dfPop.score_0.idxmax()]
        best_fitness = best_individual.score_0
        best_params = {name: best_individual[name] for name in param_names}

        logging.info(f"Optimization completed!")
        logging.info(f"Best fitness (FC correlation): {best_fitness:.4f}")
        logging.info(f"Best parameters: {best_params}")

        return evolution

    def update_parameters(self, params: Dict[str, float]):
        """
        Update network parameters.

        Parameters:
            params: Dictionary of parameters to update.
                   Can include global parameters (e.g., 'c_gl') or
                   region-specific parameters (e.g., 'node_0_Ke')
        """
        # Separate global and region-specific parameters
        for param_name, param_value in params.items():
            if param_name.startswith('node_'):
                # Region-specific parameter: format is 'node_X_paramname'
                parts = param_name.split('_', 2)
                node_idx = int(parts[1])
                actual_param_name = parts[2]

                # Update in the specific node
                if node_idx < self.n_regions:
                    self.network.nodes[node_idx].update_params({actual_param_name: param_value})
            else:
                # Global parameter: update in all nodes
                for node in self.network.nodes:
                    node.update_params({param_name: param_value})

    def get_region_info(self) -> pd.DataFrame:
        """
        Get information about all brain regions in the network.

        Returns:
            DataFrame with region information
        """
        info = []
        for i, (idx, name) in enumerate(zip(self.region_indices, self.region_names)):
            info_dict = {
                'index': i,
                'aal_index': idx,
                'name': name,
            }
            if self.region_coords is not None:
                info_dict['x'] = self.region_coords[i][0]
                info_dict['y'] = self.region_coords[i][1]
                info_dict['z'] = self.region_coords[i][2]
            info.append(info_dict)

        return pd.DataFrame(info)

    def save_results(self, output_file: str, include_timeseries: bool = False):
        """
        Save network results to file.

        Parameters:
            output_file: Output file path (.npz format)
            include_timeseries: Whether to save full time series (can be large)
        """
        results = {
            'connectivity_matrix': self.connectivity_matrix,
            'delay_matrix': self.delay_matrix,
            'region_names': self.region_names,
            'region_indices': self.region_indices,
        }

        if self.empirical_fc is not None:
            results['empirical_fc'] = self.empirical_fc

        if hasattr(self.network, 'outputs'):
            results['simulated_fc'] = func.fc(self.network.outputs.r_mean_EXC)
            if include_timeseries:
                results['firing_rates_exc'] = self.network.outputs.r_mean_EXC
                results['firing_rates_inh'] = self.network.outputs.r_mean_INH

        np.savez(output_file, **results)
        logging.info(f"Results saved to {output_file}")


def create_example_fc_matrix(n_regions: int = 116, output_file: str = "example_fc_116x116.csv"):
    """
    Create an example functional connectivity matrix for testing.

    This generates a synthetic FC matrix with realistic properties:
    - Values between -1 and 1
    - Symmetric
    - Diagonal = 1
    - Some spatial structure (nearby regions more correlated)

    Parameters:
        n_regions: Number of brain regions
        output_file: Output CSV file path
    """
    logging.info(f"Creating example FC matrix ({n_regions}x{n_regions})...")

    # Create random correlation matrix
    np.random.seed(42)  # For reproducibility

    # Generate random symmetric matrix
    fc = np.random.randn(n_regions, n_regions)
    fc = (fc + fc.T) / 2  # Make symmetric

    # Add some structure (regions close in index are more correlated)
    for i in range(n_regions):
        for j in range(n_regions):
            distance = abs(i - j)
            fc[i, j] += 2.0 * np.exp(-distance / 10.0)  # Spatial decay

    # Normalize to correlation matrix
    D = np.sqrt(np.diag(fc))
    fc = fc / np.outer(D, D)

    # Ensure values are in [-1, 1]
    fc = np.clip(fc, -1, 1)

    # Ensure diagonal is 1
    np.fill_diagonal(fc, 1.0)

    # Save to CSV
    pd.DataFrame(fc).to_csv(output_file, index=False, header=False)
    logging.info(f"Example FC matrix saved to {output_file}")

    return fc


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create example FC matrix
    create_example_fc_matrix(n_regions=116, output_file="example_fc_116x116.csv")
