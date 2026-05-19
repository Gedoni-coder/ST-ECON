"""
Universal Automated Neuro-Symbolic Hosting Engine
==================================================
Intelligence as a Service (IaaS)

This is the SINGLE unified platform. There is no separate "API Gateway."
The Engine IS the platform. The API IS the Engine's mouth.

Every endpoint is dynamically driven by what the Engine has processed.
Nothing is hardcoded. The API reflects the Engine's current living state.
"""

import sys
import os
import json
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Core engine modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from auto_ingestion import UniversalKnowledgeGraphCompiler
from dynamic_gnn_builder import DynamicTGATBuilder
from orchestrator import UniversalNeuroSymbolicOrchestrator
from feedback_loop import WeakSupervisionEngine
from tenant_registry import TenantRegistry
from transfusion_accelerator import generate_acceleration_report, EinsumCascadeDAG, DPipeScheduler
from llm_client import llm_client

# ──────────────────────────────────────────────
# GLOBAL ENGINE STATE
# The platform's living memory. Every project
# uploaded by every tenant lives here.
# ──────────────────────────────────────────────
registry = TenantRegistry()

# ──────────────────────────────────────────────
# REQUEST SCHEMAS
# ──────────────────────────────────────────────
class RegisterRequest(BaseModel):
    org_name: str

class IngestRequest(BaseModel):
    project_name: str
    knowledge_base_text: str

class QueryRequest(BaseModel):
    target_node: str
    k_hops: int = 2

class StreamSignalRequest(BaseModel):
    node_name: str
    shock_value: float

class FeedbackRequest(BaseModel):
    node: str
    action_taken: str
    initial_threat: float
    realized_outcome: float

# ──────────────────────────────────────────────
# AUTH DEPENDENCY
# ──────────────────────────────────────────────
def authenticate(x_api_key: str = Header(...)):
    org_id = registry.validate_key(x_api_key)
    if not org_id:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    return org_id

# ──────────────────────────────────────────────
# THE UNIFIED PLATFORM
# ──────────────────────────────────────────────
app = FastAPI(
    title="Universal Neuro-Symbolic Intelligence Engine",
    description="Intelligence as a Service. Upload any Knowledge Base. The Engine auto-generates graphs, configures GNNs, and serves dynamic intelligence through this API.",
    version="1.0.0"
)

@app.get("/health")
async def health():
    live_projects = sum(1 for p in registry.projects.values() if p["status"] == "live")
    return {
        "status": "operational",
        "tenants": len(registry.tenants),
        "total_projects": len(registry.projects),
        "live_projects": live_projects
    }

# ──────────────────────────────────────────────
# 1. TENANT REGISTRATION
# ──────────────────────────────────────────────
@app.post("/register")
async def register_tenant(req: RegisterRequest):
    result = registry.register_tenant(req.org_name)
    return {
        "message": f"Tenant '{req.org_name}' registered.",
        "org_id": result["org_id"],
        "api_key": result["api_key"]
    }

# ──────────────────────────────────────────────
# 2. INGEST A KNOWLEDGE BASE (Create a Project)
#    This is where the Engine does its work.
#    The API simply triggers the Engine and
#    stores whatever the Engine produces.
# ──────────────────────────────────────────────
@app.post("/projects/ingest")
async def ingest_knowledge_base(req: IngestRequest, org_id: str = Depends(authenticate)):
    # Create a new project for this tenant
    project_id = registry.register_project(org_id, req.project_name)
    
    # ── ENGINE STAGE 1: Auto-Ingestion ──
    compiler = UniversalKnowledgeGraphCompiler()
    nx_graph = await compiler.parse_knowledge_base(req.knowledge_base_text, llm_client=llm_client)
    graph_json = compiler.get_graph_schema_json()
    actions = compiler.extracted_actions
    
    # ── ENGINE STAGE 2: Dynamic GNN Auto-Configuration ──
    gnn = DynamicTGATBuilder(graph_json)
    
    # ── ENGINE STAGE 4: Weak Supervision Initialization ──
    supervision = WeakSupervisionEngine(list(actions))
    
    # Store the live engine state inside the project
    registry.set_project_engine(project_id, {
        "compiler": compiler,
        "nx_graph": nx_graph,
        "gnn": gnn,
        "supervision": supervision,
        "actions": actions,
        "graph_json": graph_json
    })
    
    return {
        "project_id": project_id,
        "project_name": req.project_name,
        "status": "live",
        "auto_generated_graph": {
            "nodes": gnn.node_names,
            "num_edges": nx_graph.number_of_edges(),
        },
        "auto_extracted_actions": list(actions)
    }

# ──────────────────────────────────────────────
# 3. QUERY A PROJECT (The Intelligence Output)
#    The API asks the Engine: "What intelligence
#    do you have for this query?" The Engine
#    responds with a structured TaCIE prompt +
#    counterfactual analysis.
# ──────────────────────────────────────────────
@app.post("/projects/{project_id}/query")
async def query_project(project_id: str, req: QueryRequest, org_id: str = Depends(authenticate)):
    project = registry.get_project(project_id)
    if not project or project["status"] != "live":
        raise HTTPException(status_code=404, detail="Project not found or not yet ingested.")
    
    engine = project["engine_state"]
    gnn_states = engine["gnn"].get_network_state()
    
    # ── ENGINE STAGE 3: Orchestration ──
    orchestrator = UniversalNeuroSymbolicOrchestrator(
        nx_graph=engine["nx_graph"],
        gnn_states=gnn_states,
        available_actions=engine["actions"],
        action_weights=engine["supervision"].action_weights
    )
    
    try:
        tacie_prompt = orchestrator.generate_universal_tacie_prompt(req.target_node)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Extract subgraph data for the response
    subgraph_data = orchestrator.extract_subgraph(req.target_node, req.k_hops)
    
    # ── TRANSFUSION ACCELERATION (Paper 2) ──
    num_nodes = engine["gnn"].num_nodes
    accel = generate_acceleration_report(seq_len=max(32, num_nodes * 16), d_model=64, n_heads=4)
    
    # ── LLM ADVISORY GENERATION ──
    action_confidence = engine["supervision"].get_action_confidence()
    llm_result = await llm_client.generate(
        tacie_prompt=tacie_prompt,
        target_node=req.target_node,
        threat_level=subgraph_data["threat_states"].get(req.target_node, 0.0),
        actions=action_confidence
    )

    return {
        "project_id": project_id,
        "target_node": req.target_node,
        "current_threat": subgraph_data["threat_states"].get(req.target_node, 0.0),
        "local_network": subgraph_data["linearized_text"],
        "threat_map": subgraph_data["threat_states"],
        "counterfactual_actions": action_confidence,
        "advisory": llm_result["advisory"],
        "advisory_model": llm_result["model"],
        "advisory_backend": llm_result["backend"],
        "tacie_prompt_for_llm": tacie_prompt,
        "transfusion_acceleration": {
            "dpipe_speedup": accel["dpipe_scheduling"]["speedup"],
            "dpipe_pipelined_latency_us": accel["dpipe_scheduling"]["pipelined_latency_us"],
            "pe_2d_utilization_pct": accel["dpipe_scheduling"]["pe_2d_util"],
            "pe_1d_utilization_pct": accel["dpipe_scheduling"]["pe_1d_util"],
            "tileseek_tile_p": accel["tileseek_tiling"]["tile_p"],
            "tileseek_sram_util_pct": accel["tileseek_tiling"]["util"],
            "attention_memory_reduction_x": accel["one_pass_attention"]["memory_reduction"],
            "attention_numerical_error": accel["one_pass_attention"]["max_diff"],
            "dram_savings_pct": accel["inter_layer_propagation"]["dram_savings_pct"]
        }
    }

# ──────────────────────────────────────────────
# 4. STREAM A LIVE SIGNAL INTO A PROJECT
#    The Engine accepts real-time perturbations
#    and propagates them through its GNN.
# ──────────────────────────────────────────────
@app.post("/projects/{project_id}/stream")
async def stream_signal(project_id: str, req: StreamSignalRequest, org_id: str = Depends(authenticate)):
    project = registry.get_project(project_id)
    if not project or project["status"] != "live":
        raise HTTPException(status_code=404, detail="Project not found or not yet ingested.")
    
    engine = project["engine_state"]
    engine["gnn"].stream_continuous_signal(req.node_name, req.shock_value)
    
    # Save the updated GNN signal state back to SQLite
    registry.set_project_engine(project_id, engine)
    
    return {
        "project_id": project_id,
        "signal_injected": {"node": req.node_name, "value": req.shock_value},
        "updated_network_state": engine["gnn"].get_network_state()
    }

# ──────────────────────────────────────────────
# 5. SUBMIT FEEDBACK (Weak Supervision Loop)
#    The external app reports real-world outcomes.
#    The Engine learns which actions actually work.
# ──────────────────────────────────────────────
@app.post("/projects/{project_id}/feedback")
async def submit_feedback(project_id: str, req: FeedbackRequest, org_id: str = Depends(authenticate)):
    project = registry.get_project(project_id)
    if not project or project["status"] != "live":
        raise HTTPException(status_code=404, detail="Project not found or not yet ingested.")
    
    engine = project["engine_state"]
    engine["supervision"].submit_retrospective_outcome(
        node=req.node,
        action_taken=req.action_taken,
        initial_threat=req.initial_threat,
        realized_outcome=req.realized_outcome
    )
    
    # Save the updated learning weights back to SQLite
    registry.set_project_engine(project_id, engine)
    
    return {
        "project_id": project_id,
        "updated_action_confidence": engine["supervision"].get_action_confidence()
    }

# ──────────────────────────────────────────────
# 6. LIST ALL LIVE PROJECTS (Dynamic Discovery)
#    External apps can discover what intelligence
#    is currently available on the platform.
# ──────────────────────────────────────────────
@app.get("/projects")
async def list_projects(org_id: str = Depends(authenticate)):
    tenant_projects = registry.get_tenant_projects(org_id)
    
    projects = []
    for tp in tenant_projects:
        pid = tp["project_id"]
        p = registry.get_project(pid)
        if p:
            engine = p.get("engine_state")
            projects.append({
                "project_id": pid,
                "project_name": p["project_name"],
                "status": p["status"],
                "nodes": engine["gnn"].node_names if engine else [],
                "actions": list(engine["actions"]) if engine else []
            })
    return {"org_id": org_id, "projects": projects}

# ──────────────────────────────────────────────
# 7. ACCELERATION PROFILING
#    Exposes the full TransFusion report for a
#    project's graph topology.
# ──────────────────────────────────────────────
@app.get("/projects/{project_id}/acceleration")
async def acceleration_profile(project_id: str, org_id: str = Depends(authenticate)):
    project = registry.get_project(project_id)
    if not project or project["status"] != "live":
        raise HTTPException(status_code=404, detail="Project not found or not yet ingested.")
    engine = project["engine_state"]
    num_nodes = engine["gnn"].num_nodes
    report = generate_acceleration_report(seq_len=max(32, num_nodes * 16), d_model=64, n_heads=4)
    return {"project_id": project_id, "transfusion_report": report}


@app.on_event("shutdown")
def shutdown_event():
    print("[Server] Terminating active engine sub-runtimes...")
    llm_client.shutdown()


# ──────────────────────────────────────────────
# LAUNCH
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
