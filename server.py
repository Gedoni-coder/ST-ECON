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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
import numpy as np

# Try importing torch-based edge lab modules, fallback to NumPy twins if PyTorch is not installed
try:
    import torch
    import torch.nn as nn
    from experiments.mas_attention import MASAttention
    from experiments.quantize_model import LinearAttentionRegressor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class MASAttentionTwin:
    def __init__(self, d_model: int = 128, n_heads: int = 4, sram_limit_bytes: int = 4096):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sram_limit_bytes = sram_limit_bytes
        # Seed for reproducible outputs
        rng = np.random.default_rng(42)
        self.q_w = rng.normal(0.0, 0.01, (d_model, d_model))
        self.k_w = rng.normal(0.0, 0.01, (d_model, d_model))
        self.v_w = rng.normal(0.0, 0.01, (d_model, d_model))
        self.out_w = rng.normal(0.0, 0.01, (d_model, d_model))

    def forward(self, x: np.ndarray, use_mixed_precision: bool = True) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        element_size = 2 if use_mixed_precision else 4
        block_size = max(8, self.sram_limit_bytes // (self.d_k * element_size * 2))
        
        Q = np.dot(x, self.q_w)
        K = np.dot(x, self.k_w)
        V = np.dot(x, self.v_w)
        
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        output = np.zeros_like(Q)
        
        for q_start in range(0, seq_len, block_size):
            q_end = min(q_start + block_size, seq_len)
            Q_block = Q[:, :, q_start:q_end, :]
            
            accum_num = np.zeros((batch_size, self.n_heads, q_end - q_start, self.d_k))
            accum_den = np.zeros((batch_size, self.n_heads, q_end - q_start, 1))
            
            for k_start in range(0, seq_len, block_size):
                k_end = min(k_start + block_size, seq_len)
                K_block = K[:, :, k_start:k_end, :]
                V_block = V[:, :, k_start:k_end, :]
                
                scores = np.matmul(Q_block, K_block.transpose(0, 1, 3, 2)) / (self.d_k ** 0.5)
                max_scores = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - max_scores)
                
                accum_num += np.matmul(exp_scores, V_block)
                accum_den += np.sum(exp_scores, axis=-1, keepdims=True)
                
            output[:, :, q_start:q_end, :] = accum_num / (accum_den + 1e-9)
            
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return np.dot(output, self.out_w)

class LinearAttentionRegressorTwin:
    def __init__(self, d_model: int = 128):
        rng = np.random.default_rng(42)
        self.fc1_w = rng.normal(0.0, 0.01, (d_model, 256))
        self.fc1_b = np.zeros(256)
        self.fc2_w = rng.normal(0.0, 0.01, (256, 32))
        self.fc2_b = np.zeros(32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x1 = np.maximum(0, np.dot(x, self.fc1_w) + self.fc1_b)
        return np.dot(x1, self.fc2_w) + self.fc2_b

def run_edge_inference_pipeline(H: np.ndarray, sram_limit_bytes: int = 4096, use_mixed_precision: bool = True, precision: str = "FP16"):
    num_nodes = len(H)
    precision_upper = precision.upper()
    if precision_upper == "FP32":
        element_size = 4
        use_mixed_precision = False
    elif precision_upper == "INT8":
        element_size = 1
        use_mixed_precision = True
    else: # FP16
        element_size = 2
        use_mixed_precision = True
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    
    x_np = np.zeros((1, num_nodes, d_model), dtype=np.float32)
    for i in range(num_nodes):
        freqs = np.sin(np.arange(d_model) * (i + 1) * 0.1)
        x_np[0, i, :] = H[i] * freqs
        
    block_size = max(8, sram_limit_bytes // (d_k * element_size * 2))
    
    if HAS_TORCH:
        torch.manual_seed(42)
        mas_model = MASAttention(d_model=d_model, n_heads=n_heads, sram_limit_bytes=sram_limit_bytes)
        reg_model = LinearAttentionRegressor(d_model=d_model)
        
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            attn_out = mas_model(x_tensor, use_mixed_precision=use_mixed_precision)
            reg_out = reg_model(attn_out)
        
        reg_out_np = reg_out.numpy()
        framework = "PyTorch"
    else:
        mas_twin = MASAttentionTwin(d_model=d_model, n_heads=n_heads, sram_limit_bytes=sram_limit_bytes)
        reg_twin = LinearAttentionRegressorTwin(d_model=d_model)
        
        attn_out_np = mas_twin.forward(x_np, use_mixed_precision=use_mixed_precision)
        reg_out_np = reg_twin.forward(attn_out_np)
        framework = "NumPy (Mathematical Twin Fallback)"
        
    original_dram_activation_bytes = int(3 * 1 * num_nodes * d_model * element_size + 1 * n_heads * num_nodes * num_nodes * element_size)
    
    peak_sram_bytes = int(
        (n_heads * block_size * d_k * element_size) + 
        (2 * n_heads * block_size * d_k * element_size) + 
        (n_heads * block_size * block_size * element_size) +
        (n_heads * block_size * d_k * element_size) +
        (n_heads * block_size * 1 * element_size)
    )
    
    memory_reduction_ratio = float(original_dram_activation_bytes / max(1.0, peak_sram_bytes))
    register_footprint_units = int(n_heads * (d_k + block_size) * (2 if use_mixed_precision else 4))
    
    mean_risk_metrics = np.mean(reg_out_np[0], axis=0).tolist()
    
    return {
        "framework": framework,
        "sequence_length": num_nodes,
        "block_size": block_size,
        "peak_sram_usage_bytes": peak_sram_bytes,
        "register_footprint_units": register_footprint_units,
        "memory_reduction_ratio": round(memory_reduction_ratio, 2),
        "original_dram_activation_bytes": original_dram_activation_bytes,
        "streamed_sram_activation_bytes": peak_sram_bytes,
        "regressor_output_risk_metrics": mean_risk_metrics
    }

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

class EdgeInferenceRequest(BaseModel):
    sram_limit_bytes: Optional[int] = 4096
    use_mixed_precision: Optional[bool] = True
    precision: Optional[str] = "FP16"

class SimulateTemporalRequest(BaseModel):
    node_name: str
    shock_value: float
    steps: Optional[int] = 20
    dt: Optional[float] = 1.0

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

# Enable CORS for UI interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI
ui_path = os.path.join(os.path.dirname(__file__), "ui")
os.makedirs(ui_path, exist_ok=True)
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

@app.get("/health")
async def health():
    stats = registry.get_platform_stats()
    return {
        "status": "operational",
        "tenants": stats["tenants"],
        "total_projects": stats["total_projects"],
        "live_projects": stats["live_projects"]
    }

@app.get("/")
async def root():
    return FileResponse(os.path.join(ui_path, "index.html"))
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
    
    # ── FORMAL SYMBOLIC PRIORS & CONSTRAINT SATISFIABILITY PROOFS ──
    from symbolic_verifier import SymbolicPriorVerifier
    base_threat = subgraph_data["threat_states"].get(req.target_node, 0.0)
    verifier = SymbolicPriorVerifier(engine["nx_graph"], safety_threshold=3.5, resource_budget=5.0)
    verify_report = verifier.verify_action_safety(req.target_node, base_threat, engine["supervision"].action_weights)

    # ── LLM ADVISORY GENERATION ──
    action_confidence = engine["supervision"].get_action_confidence()
    llm_result = await llm_client.generate(
        tacie_prompt=tacie_prompt,
        target_node=req.target_node,
        threat_level=base_threat,
        actions=action_confidence
    )

    return {
        "project_id": project_id,
        "target_node": req.target_node,
        "current_threat": base_threat,
        "local_network": subgraph_data["linearized_text"],
        "threat_map": subgraph_data["threat_states"],
        "counterfactual_actions": action_confidence,
        "advisory": llm_result["advisory"],
        "advisory_model": llm_result["model"],
        "advisory_backend": llm_result["backend"],
        "tacie_prompt_for_llm": tacie_prompt,
        "verify_report": verify_report,
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

# ──────────────────────────────────────────────
# 8. EDGE INFERENCE SIMULATION
#    Exposes SRAM block-streaming attention memory metrics
#    and Regressor risk metrics for living GNN states.
# ──────────────────────────────────────────────
@app.post("/projects/{project_id}/edge_inference")
async def edge_inference_simulation(project_id: str, req: EdgeInferenceRequest, org_id: str = Depends(authenticate)):
    project = registry.get_project(project_id)
    if not project or project["status"] != "live":
        raise HTTPException(status_code=404, detail="Project not found or not yet ingested.")
        
    engine = project["engine_state"]
    H = engine["gnn"].H
    
    try:
        report = run_edge_inference_pipeline(
            H=H,
            sram_limit_bytes=req.sram_limit_bytes,
            use_mixed_precision=req.use_mixed_precision,
            precision=req.precision or "FP16"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edge inference simulation failed: {str(e)}")
        
    return {
        "project_id": project_id,
        "edge_inference_simulation": report
    }

# ──────────────────────────────────────────────
# 9. TEMPORAL CONTINUOUS SIMULATION (RK4)
#    Simulates threat propagation over time
# ──────────────────────────────────────────────
@app.post("/projects/{project_id}/simulate_temporal")
async def simulate_temporal(project_id: str, req: SimulateTemporalRequest, org_id: str = Depends(authenticate)):
    project = registry.get_project(project_id)
    if not project or project["status"] != "live":
        raise HTTPException(status_code=404, detail="Project not found or not yet ingested.")
    
    engine = project["engine_state"]
    gnn = engine["gnn"]
    
    # Run the continuous RK4 integration step-by-step
    sim_H = gnn.H.copy()
    sim_W_edge = gnn.W_edge.copy()
    
    # Inject shock to the starting node
    if req.node_name in gnn.node_idx:
        idx = gnn.node_idx[req.node_name]
        sim_H[idx] += req.shock_value
        
    timeline = []
    # Step 0
    timeline.append({name: float(sim_H[idx]) for name, idx in gnn.node_idx.items()})
    
    # Let's import the Numba ODE solver function from dynamic_gnn_builder
    from dynamic_gnn_builder import _numba_propagate_ode
    
    for _ in range(req.steps):
        sim_H, sim_W_edge = _numba_propagate_ode(sim_H, sim_W_edge, gnn.A, dt=req.dt, alpha=gnn.alpha)
        timeline.append({name: float(sim_H[idx]) for name, idx in gnn.node_idx.items()})
        
    return {
        "project_id": project_id,
        "shocked_node": req.node_name,
        "shock_value": req.shock_value,
        "timeline": timeline
    }


@app.on_event("shutdown")
def shutdown_event():
    print("[Server] Terminating active engine sub-runtimes...")
    llm_client.shutdown()


# ──────────────────────────────────────────────
# LAUNCH
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
