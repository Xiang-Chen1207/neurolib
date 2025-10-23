"""
Simple test script for AAL-ALN network implementation
"""

import sys
import os
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add neurolib to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from neurolib.models.aal_aln_network import AALALNNetwork, create_example_fc_matrix
    logging.info("✓ Successfully imported AALALNNetwork")
except Exception as e:
    logging.error(f"✗ Failed to import AALALNNetwork: {e}")
    sys.exit(1)

def test_basic_creation():
    """Test basic network creation"""
    logging.info("\n--- Test 1: Basic Network Creation ---")
    try:
        # Create simple connectivity
        n_regions = 116
        connectivity = np.eye(n_regions) * 0.1

        # Create network
        network = AALALNNetwork(
            connectivity_matrix=connectivity,
            n_regions=n_regions
        )

        logging.info(f"✓ Network created with {network.n_regions} regions")
        logging.info(f"✓ Region names sample: {network.region_names[:3]}")
        return True
    except Exception as e:
        logging.error(f"✗ Test failed: {e}")
        return False

def test_fc_loading():
    """Test FC matrix loading"""
    logging.info("\n--- Test 2: FC Matrix Loading ---")
    try:
        # Create example FC
        fc_file = "test_fc_116.csv"
        create_example_fc_matrix(n_regions=116, output_file=fc_file)
        logging.info(f"✓ Created example FC matrix: {fc_file}")

        # Create network
        connectivity = np.random.rand(116, 116) * 0.2
        connectivity = (connectivity + connectivity.T) / 2
        np.fill_diagonal(connectivity, 0)

        network = AALALNNetwork(connectivity_matrix=connectivity, n_regions=116)

        # Load FC
        fc = network.load_empirical_fc(fc_file)
        logging.info(f"✓ Loaded FC matrix: shape {fc.shape}")
        logging.info(f"✓ FC value range: [{fc.min():.3f}, {fc.max():.3f}]")

        # Clean up
        if os.path.exists(fc_file):
            os.remove(fc_file)

        return True
    except Exception as e:
        logging.error(f"✗ Test failed: {e}")
        return False

def test_simulation():
    """Test basic simulation"""
    logging.info("\n--- Test 3: Basic Simulation ---")
    try:
        # Create network
        n_regions = 10  # Use small number for quick test
        connectivity = np.random.rand(n_regions, n_regions) * 0.2
        connectivity = (connectivity + connectivity.T) / 2
        np.fill_diagonal(connectivity, 0)

        network = AALALNNetwork(connectivity_matrix=connectivity, n_regions=n_regions)

        # Run short simulation
        logging.info("Running simulation (this may take a minute)...")
        results = network.run_simulation(duration=1000, dt=0.1)  # 1 second

        logging.info(f"✓ Simulation completed")
        logging.info(f"✓ Firing rates shape: {results['firing_rates_exc'].shape}")
        logging.info(f"✓ Simulated FC shape: {results['simulated_fc'].shape}")

        # Check values are reasonable
        firing_rates = results['firing_rates_exc']
        if np.any(np.isnan(firing_rates)):
            logging.warning("⚠ Warning: NaN values in firing rates")
        else:
            logging.info(f"✓ Firing rates range: [{firing_rates.min():.2f}, {firing_rates.max():.2f}] Hz")

        return True
    except Exception as e:
        logging.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_region_specific_params():
    """Test region-specific parameters"""
    logging.info("\n--- Test 4: Region-Specific Parameters ---")
    try:
        n_regions = 10
        connectivity = np.eye(n_regions) * 0.1

        # Create region-specific parameters
        exc_params_list = []
        for i in range(n_regions):
            params = {'Ke': 800.0 + i * 10, 'c_gl': 0.4 + i * 0.01}
            exc_params_list.append(params)

        network = AALALNNetwork(
            connectivity_matrix=connectivity,
            n_regions=n_regions,
            exc_params=exc_params_list
        )

        logging.info(f"✓ Created network with region-specific parameters")
        logging.info(f"✓ First region Ke: {network.exc_params[0]['Ke']}")
        logging.info(f"✓ Last region Ke: {network.exc_params[-1]['Ke']}")

        return True
    except Exception as e:
        logging.error(f"✗ Test failed: {e}")
        return False

def main():
    """Run all tests"""
    logging.info("="*60)
    logging.info("AAL-ALN Network Implementation Tests")
    logging.info("="*60)

    tests = [
        test_basic_creation,
        test_fc_loading,
        test_region_specific_params,
        test_simulation,  # This one takes longer
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    # Summary
    logging.info("\n" + "="*60)
    logging.info("Test Summary")
    logging.info("="*60)
    passed = sum(results)
    total = len(results)
    logging.info(f"Passed: {passed}/{total}")

    if passed == total:
        logging.info("✓ All tests passed!")
        return 0
    else:
        logging.error(f"✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
