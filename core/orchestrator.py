import networkx as nx
import copy
from typing import Dict, List, Set

class UniversalNeuroSymbolicOrchestrator:
    def __init__(self, nx_graph: nx.DiGraph, gnn_states: Dict[str, float], available_actions: Set[str], action_weights: Dict[str, float] = None):
        """
        Stage 3: Neuro-Symbolic Orchestration & Counterfactuals.
        Bridges the mathematical GNN output to the natural language LLM.
        """
        self.graph = nx_graph
        self.gnn_states = gnn_states
        self.actions = list(available_actions)
        self.action_weights = action_weights or {}

    def extract_subgraph(self, target_node: str, k_hops: int = 2) -> dict:
        """
        Prunes the massive universal graph into a highly focused Edge-Induced Subgraph
        around the target node, reducing LLM context window load.
        """
        if not self.graph.has_node(target_node):
            raise ValueError(f"Target node {target_node} not in graph.")
            
        print(f"[Orchestrator] Extracting {k_hops}-hop subgraph around '{target_node}'...")
        
        # Extract k-hop neighborhood (both predecessors and successors)
        nodes_to_keep = {target_node}
        current_layer = {target_node}
        
        for _ in range(k_hops):
            next_layer = set()
            for node in current_layer:
                next_layer.update(self.graph.predecessors(node))
                next_layer.update(self.graph.successors(node))
            nodes_to_keep.update(next_layer)
            current_layer = next_layer
            
        subgraph = self.graph.subgraph(nodes_to_keep)
        
        # Serialize subgraph for the LLM
        linearized_context = []
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation_type", "connects to")
            linearized_context.append(f"{u} {rel} {v}.")
            
        return {
            "nodes": list(nodes_to_keep),
            "linearized_text": " ".join(linearized_context),
            "threat_states": {n: self.gnn_states.get(n, 0.0) for n in nodes_to_keep}
        }

    def run_counterfactual_simulation(self, target_node: str, base_threat: float) -> str:
        """
        Optimized Counterfactual Simulation Engine.
        Tests all actions in parallel using vectorized matrix multipliers, calculating
        threat reduction multipliers dynamically from live learned action weights.
        """
        import numpy as np
        print("[Orchestrator] Running Vectorized Counterfactual Simulations...")
        
        # Pre-compile action coefficients using numpy vector representation
        num_actions = len(self.actions)
        multipliers = np.ones(num_actions, dtype=np.float64)
        
        for i, action in enumerate(self.actions):
            # Fetch the learned weight/confidence from Weak Supervision loop (default: 1.0)
            learned_weight = self.action_weights.get(action, 1.0)
            # Dynamic reduction formula: higher learned weight means greater threat reduction (smaller multiplier)
            # Range: if weight=1.0 -> mult=0.5. If weight=4.0 -> mult=0.20. If weight=0.1 -> mult=0.91
            multipliers[i] = 1.0 / (1.0 + max(0.001, learned_weight))
                
        # Vectorized scaling of base threat across all actions simultaneously
        simulated_threats = base_threat * multipliers
        improvements = base_threat - simulated_threats
        
        # Format the output sequentially for the LLM prompt context
        simulation_results = [
            f"- If action '{self.actions[i]}' is taken, the simulated threat score at {target_node} reduces from {base_threat:.2f} to {simulated_threats[i]:.2f} (Improvement: {improvements[i]:.2f})."
            for i in range(num_actions)
        ]
            
        return "\n".join(simulation_results)

    def generate_universal_tacie_prompt(self, target_node: str) -> str:
        """
        Generates the Task-Centred Instruction (TaCIE) Prompt dynamically 
        based on the auto-generated graph, counterfactuals, and symbolic verifier proofs.
        """
        # 1. Get pruned subgraph
        subgraph_data = self.extract_subgraph(target_node)
        base_threat = subgraph_data["threat_states"][target_node]
        
        # 2. Run counterfactuals
        counterfactual_text = self.run_counterfactual_simulation(target_node, base_threat)
        
        # 3. Formally verify safety constraints using SMT / Algebraic verifier
        from symbolic_verifier import SymbolicPriorVerifier
        verifier = SymbolicPriorVerifier(self.graph, safety_threshold=3.5, resource_budget=5.0)
        verify_report = verifier.verify_action_safety(target_node, base_threat, self.action_weights)
        proof_log_text = "\n".join([f"  - {line}" for line in verify_report["proof_log"]])
        
        # 4. Assemble TaCIE prompt with Formal Proof embeddings
        prompt = f"""
[SYSTEM: UNIVERSAL NEURO-SYMBOLIC ADVISOR]
You are a strategic intelligence AI. You must analyze the following extracted Knowledge Graph subgraph and provide operational advice.

[STAGE 1: GRAPH TOPOLOGY & DYNAMIC THREAT STATE]
Target Entity: {target_node}
Current Threat Level: {base_threat:.3f}

Local Trade Network (Linearized):
{subgraph_data['linearized_text']}

Threat Exposure of Local Network:
{subgraph_data['threat_states']}

[STAGE 2: COUNTERFACTUAL GNN SIMULATIONS]
Our temporal math engine has simulated the available mitigation actions:
{counterfactual_text}

[STAGE 3: FORMAL SYMBOLIC PRIORS & CONSTRAINT SATISFIABILITY PROOFS]
The verification engine successfully solved the logical boundary conditions for safety and resource allocation constraints:
- Solver Engine: {verify_report['solver_engine']}
- Safety Saturation Satisfied: {verify_report['safety_satisfied']}
- Optimal Action Allocation Ratios: {verify_report['optimized_commitment_ratios']}
- Analytical Proof Log:
{proof_log_text}

[STAGE 4: BOOSTING OF THOUGHTS REASONING]
Perform a step-by-step analysis:
1. Threat Translation: How does the threat propagate through the linearized network above to reach {target_node}?
2. Action Evaluation: Based on the counterfactual simulations and SMT/Algebraic solver commitments, which actions are formally proven safe?
3. Self-Critique: Are there secondary risks to taking these actions?
4. Final Recommendation: Provide the final mitigation strategy.
"""
        return prompt.strip()

if __name__ == "__main__":
    # Mocking the pipeline output from Stage 1 and 2
    mock_graph = nx.DiGraph()
    mock_graph.add_edge("GLOBALSHIPPINGLANE", "LITHIUMMINE", relation_type="affects")
    mock_graph.add_edge("LITHIUMMINE", "BATTERYFACTORY", relation_type="supplies")
    mock_graph.add_edge("BATTERYFACTORY", "AUTOMOTIVEPLANT", relation_type="supplies")
    
    mock_gnn_states = {
        "GLOBALSHIPPINGLANE": 5.0,
        "LITHIUMMINE": 3.2,
        "BATTERYFACTORY": 1.8,
        "AUTOMOTIVEPLANT": 2.7
    }
    
    mock_actions = {"increase_local_inventory", "forward_contracts"}
    mock_weights = {"increase_local_inventory": 1.5, "forward_contracts": 0.25}
    orchestrator = UniversalNeuroSymbolicOrchestrator(mock_graph, mock_gnn_states, mock_actions, mock_weights)
    prompt = orchestrator.generate_universal_tacie_prompt("AUTOMOTIVEPLANT")
    
    print("\n--- Auto-Generated Universal TaCIE Prompt with SMT Proofs ---")
    print(prompt)
