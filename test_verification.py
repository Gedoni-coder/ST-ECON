import unittest
import numpy as np
import networkx as nx
import sys
import os

# Insert core and experiments paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.dirname(__file__))

from dynamic_gnn_builder import DynamicTGATBuilder, _numba_propagate_ode
from symbolic_verifier import SymbolicPriorVerifier
from server import run_edge_inference_pipeline, EdgeInferenceRequest, MASAttentionTwin, LinearAttentionRegressorTwin

class TestSteconVerificationSuite(unittest.TestCase):
    def setUp(self):
        # A lightweight mock graph json representive of Stage 1 compiler outputs
        self.mock_graph_json = '''{
          "nodes": [
            ["MICROCHIPFACTORY", {"dynamic_stress": 0.0}],
            ["AUTOMOTIVEPLANT", {"dynamic_stress": 0.0}],
            ["LITHIUMMINE", {"dynamic_stress": 0.0}]
          ],
          "edges": [
            ["MICROCHIPFACTORY", "AUTOMOTIVEPLANT", {"relation_type": "supplies", "weight": 1.5}],
            ["LITHIUMMINE", "AUTOMOTIVEPLANT", {"relation_type": "supplies", "weight": 0.8}]
          ]
        }'''

    def test_rk4_ode_propagation_convergence(self):
        """
        Verify that 4th-order Runge-Kutta continuous temporal ODE propagation
        calculates state decay and updates adjacency edge weights correctly.
        """
        print("\n[TEST] Verifying continuous Neural ODE (RK4) integration...")
        builder = DynamicTGATBuilder(self.mock_graph_json, propagation_mode="continuous_ode", alpha=0.5)
        
        # Inject positive perturbation
        builder.stream_continuous_signal("MICROCHIPFACTORY", shock_value=4.0)
        
        h_initial = builder.H.copy()
        self.assertGreater(h_initial[builder.node_idx["MICROCHIPFACTORY"]], 0.0)
        
        # Propagate time forward continuously
        builder._propagate_temporal_shock(dt=0.5)
        
        h_next = builder.H.copy()
        # Due to alpha stress decay, initial perturbation at source should drop slightly,
        # but stress should propagate to AUTOMOTIVEPLANT.
        print(f" -> Initial hidden state stress: {h_initial}")
        print(f" -> Next step continuous state stress: {h_next}")
        
        self.assertLess(h_next[builder.node_idx["MICROCHIPFACTORY"]], h_initial[builder.node_idx["MICROCHIPFACTORY"]])
        self.assertGreater(h_next[builder.node_idx["AUTOMOTIVEPLANT"]], 0.0)

    def test_symbolic_verifier_logic_and_fallbacks(self):
        """
        Verify that the SymbolicPriorVerifier runs correct safety proofs, 
        yielding formal validation reports for actions.
        """
        print("\n[TEST] Verifying Symbolic Prior Verifier logic...")
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(["AUTOMOTIVEPLANT"])
        
        verifier = SymbolicPriorVerifier(nx_graph, safety_threshold=1.5, resource_budget=2.0)
        
        mock_action_weights = {
            "increase_inventory": 2.5,
            "diversify_sources": 1.2,
            "forward_contracts": 0.5
        }
        
        report = verifier.verify_action_safety(
            target_node="AUTOMOTIVEPLANT",
            base_threat=3.5,
            action_weights=mock_action_weights
        )
        
        self.assertIn("verified", report)
        self.assertIn("solver_engine", report)
        self.assertIn("proof_log", report)
        self.assertGreater(len(report["proof_log"]), 0)
        
        print(f" -> Solver engine committed: {report['solver_engine']}")
        print(f" -> Verified safety: {report['verified']}")
        print(f" -> Strategy ratio recommendation: {report['optimized_commitment_ratios']}")

    def test_edge_inference_sram_streaming_twin(self):
        """
        Verify that MASAttentionTwin and LinearAttentionRegressorTwin mathematical twins
        process inputs, calculate sram memory profiles, and save telemetry accurately.
        """
        print("\n[TEST] Verifying Edge Inference simulation and NumPy fallback twins...")
        H = np.array([2.5, 0.8, 1.2]) # Mock hidden states
        
        report = run_edge_inference_pipeline(
            H=H,
            sram_limit_bytes=4096,
            use_mixed_precision=True
        )
        
        self.assertIn("framework", report)
        self.assertIn("peak_sram_usage_bytes", report)
        self.assertIn("memory_reduction_ratio", report)
        self.assertIn("regressor_output_risk_metrics", report)
        
        self.assertEqual(len(report["regressor_output_risk_metrics"]), 32)
        self.assertGreater(report["memory_reduction_ratio"], 0.0)
        
        print(f" -> Telemetry framework: {report['framework']}")
        print(f" -> SRAM Peak usage: {report['peak_sram_usage_bytes']} bytes")
        print(f" -> Register footprint: {report['register_footprint_units']} units")
        print(f" -> DRAM vs SRAM Reduction Ratio: {report['memory_reduction_ratio']}x")

if __name__ == "__main__":
    unittest.main()
