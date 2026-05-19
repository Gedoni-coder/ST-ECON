import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from auto_ingestion import UniversalKnowledgeGraphCompiler
from dynamic_gnn_builder import DynamicTGATBuilder
from orchestrator import UniversalNeuroSymbolicOrchestrator
from feedback_loop import WeakSupervisionEngine

import asyncio

async def run_universal_pipeline():
    print("================================================================")
    print("      UNIVERSAL AUTOMATED NEURO-SYMBOLIC HOSTING ENGINE         ")
    print("================================================================\n")
    
    # ---------------------------------------------------------
    # STAGE 1: Auto-Ingestion (Raw Text -> Graph Schema)
    # ---------------------------------------------------------
    print(">>> STAGE 1: INGESTING RAW KNOWLEDGE BASE (TEXT)")
    # This is NOT Nigerian economics. It is a completely generic healthcare supply chain.
    raw_kb = """
    The overseas_pharma_lab supplies the national_medical_depot.
    The national_medical_depot distributes to the regional_hospital.
    The regional_hospital distributes to the local_clinic.
    The global_shipping_lane affects the overseas_pharma_lab.
    The ministry_of_health regulates the national_medical_depot.
    To survive shipping delays, you should stockpile_critical_medicine.
    To survive regulatory audits, you should enforce_compliance_logs.
    """
    
    nlp_engine = UniversalKnowledgeGraphCompiler()
    await nlp_engine.parse_knowledge_base(raw_kb)
    graph_schema_json = nlp_engine.get_graph_schema_json()
    extracted_actions = nlp_engine.extracted_actions
    print("\n")
    
    # ---------------------------------------------------------
    # STAGE 2: Dynamic GNN Construction & Temporal Streaming
    # ---------------------------------------------------------
    print(">>> STAGE 2: AUTO-CONFIGURING DYNAMIC T-GAT ENGINE")
    gnn_engine = DynamicTGATBuilder(graph_schema_json)
    
    print("\n--- Simulating Live Continuous Temporal Streaming ---")
    # A massive shipping delay occurs globally
    gnn_engine.stream_continuous_signal("GLOBALSHIPPINGLANE", shock_value=8.0)
    # The Ministry of Health triggers a sudden audit
    gnn_engine.stream_continuous_signal("MINISTRYOFHEALTH", shock_value=3.0)
    
    # Propagate the shocks through the auto-generated healthcare graph
    gnn_engine._propagate_temporal_shock()
    current_threat_states = gnn_engine.get_network_state()
    print("\n")

    # ---------------------------------------------------------
    # STAGE 3: Neuro-Symbolic Orchestration & Counterfactuals
    # ---------------------------------------------------------
    print(">>> STAGE 3: NEURO-SYMBOLIC ORCHESTRATION")
    # An app queries the platform for advice regarding the 'REGIONALHOSPITAL'
    target_node = "REGIONALHOSPITAL"
    
    orchestrator = UniversalNeuroSymbolicOrchestrator(
        nx_graph=nlp_engine.graph,
        gnn_states=current_threat_states,
        available_actions=extracted_actions,
        action_weights={}
    )
    
    # The platform automatically generates the TaCIE prompt, pruning the graph
    # and simulating counterfactual actions.
    final_prompt = orchestrator.generate_universal_tacie_prompt(target_node)
    
    print(f"\n--- Output Directed to Hosted LLM (vLLM) ---")
    print(final_prompt)
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(run_universal_pipeline())
