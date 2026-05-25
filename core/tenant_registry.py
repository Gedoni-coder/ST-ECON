import json
import hashlib
import secrets
import time
import os
import sqlite3
import networkx as nx
from typing import Dict, Optional

# Core imports for state reconstruction
from auto_ingestion import UniversalKnowledgeGraphCompiler
from dynamic_gnn_builder import DynamicTGATBuilder
from feedback_loop import WeakSupervisionEngine


class TenantRegistry:
    """
    Manages API key issuance, validation, and project-level access control.
    Backed by a local SQLite database to survive server restarts.
    """
    def __init__(self, db_path: str = "data/platform.db"):
        self.db_path = db_path
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize SQLite tables
        self._init_db()
        
        # In-memory cache for live active engines (reduces DB hits)
        self.active_cache: Dict[str, dict] = {}

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    org_id TEXT PRIMARY KEY,
                    org_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    total_calls INTEGER DEFAULT 0
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    hashed_key TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    FOREIGN KEY(org_id) REFERENCES tenants(org_id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    graph_json TEXT,
                    gnn_H TEXT,
                    gnn_W TEXT,
                    action_weights TEXT,
                    actions TEXT,
                    FOREIGN KEY(org_id) REFERENCES tenants(org_id)
                )
            """)
            conn.commit()

    def register_tenant(self, org_name: str) -> dict:
        org_id = hashlib.sha256(org_name.encode()).hexdigest()[:12]
        raw_key = f"nse_{secrets.token_hex(24)}"
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        
        with self._get_conn() as conn:
            c = conn.cursor()
            # Ignore if already exists, just generate new key
            c.execute(
                "INSERT OR IGNORE INTO tenants (org_id, org_name, created_at) VALUES (?, ?, ?)",
                (org_id, org_name, time.time())
            )
            c.execute(
                "INSERT INTO api_keys (hashed_key, org_id) VALUES (?, ?)",
                (hashed, org_id)
            )
            conn.commit()
            
        return {"org_id": org_id, "api_key": raw_key}

    def validate_key(self, raw_key: str) -> Optional[str]:
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT org_id FROM api_keys WHERE hashed_key = ?", (hashed,))
            row = c.fetchone()
            if row:
                org_id = row[0]
                c.execute("UPDATE tenants SET total_calls = total_calls + 1 WHERE org_id = ?", (org_id,))
                conn.commit()
                return org_id
        return None

    def register_project(self, org_id: str, project_name: str) -> str:
        project_id = hashlib.sha256(
            f"{org_id}:{project_name}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO projects (project_id, org_id, project_name, created_at, status) VALUES (?, ?, ?, ?, ?)",
                (project_id, org_id, project_name, time.time(), "pending_ingestion")
            )
            conn.commit()
            
        return project_id

    def set_project_engine(self, project_id: str, engine_state: dict):
        """Saves/updates the state of the neuro-symbolic engine in the SQLite DB."""
        # 1. Update in-memory cache
        self.active_cache[project_id] = engine_state
        
        # 2. Serialize objects to clean formats
        import numpy as np
        gnn = engine_state["gnn"]
        supervision = engine_state["supervision"]
        
        gnn_H_serialized = json.dumps(gnn.H.tolist())
        gnn_W_serialized = json.dumps(gnn.W_edge.tolist())
        action_weights_serialized = json.dumps(supervision.action_weights)
        actions_serialized = json.dumps(list(engine_state["actions"]))
        graph_json = engine_state["graph_json"]
        
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute(
                """UPDATE projects SET 
                   status = 'live',
                   graph_json = ?,
                   gnn_H = ?,
                   gnn_W = ?,
                   action_weights = ?,
                   actions = ?
                   WHERE project_id = ?""",
                (graph_json, gnn_H_serialized, gnn_W_serialized, action_weights_serialized, actions_serialized, project_id)
            )
            conn.commit()

    def get_project(self, project_id: str) -> Optional[dict]:
        """
        Loads project metadata. If the engine state is in memory, it is attached.
        If the server restarted, this method rebuilds the python objects from the DB.
        """
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT project_name, org_id, created_at, status, graph_json, gnn_H, gnn_W, action_weights, actions FROM projects WHERE project_id = ?", (project_id,))
            row = c.fetchone()
            if not row:
                return None
                
            project_name, org_id, created_at, status, graph_json, gnn_H, gnn_W, action_weights, actions = row
            
            project_data = {
                "project_name": project_name,
                "org_id": org_id,
                "created_at": created_at,
                "status": status,
                "engine_state": None
            }
            
            if status == "live":
                # Check cache first
                if project_id in self.active_cache:
                    project_data["engine_state"] = self.active_cache[project_id]
                else:
                    # RECONSTRUCT living engine from serialized DB fields
                    import numpy as np
                    print(f"[Durability] Rebuilding Engine state from DB cache for project '{project_id}'...")
                    
                    # 1. Compiler
                    compiler = UniversalKnowledgeGraphCompiler()
                    compiler.extracted_actions = set(json.loads(actions))
                    # Reconstruct NetworkX Graph from the saved custom JSON schema
                    data = json.loads(graph_json)
                    for n in data["nodes"]:
                        compiler.graph.add_node(n[0], **n[1])
                    for edge in data["edges"]:
                        compiler.graph.add_edge(edge[0], edge[1], **edge[2])
                        
                    # 2. GNN builder
                    gnn = DynamicTGATBuilder(graph_json)
                    gnn.H = np.array(json.loads(gnn_H))
                    gnn.W_edge = np.array(json.loads(gnn_W))
                    
                    # 3. Supervision feedback loop
                    supervision = WeakSupervisionEngine(json.loads(actions))
                    supervision.action_weights = json.loads(action_weights)
                    
                    engine_state = {
                        "compiler": compiler,
                        "nx_graph": compiler.graph,
                        "gnn": gnn,
                        "supervision": supervision,
                        "actions": compiler.extracted_actions,
                        "graph_json": graph_json
                    }
                    # Populate memory cache
                    self.active_cache[project_id] = engine_state
                    project_data["engine_state"] = engine_state
                    
            return project_data

    def get_tenant_projects(self, org_id: str) -> list:
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT project_id, project_name, status, created_at FROM projects WHERE org_id = ?", (org_id,))
            rows = c.fetchall()
            return [
                {
                    "project_id": r[0],
                    "project_name": r[1],
                    "status": r[2],
                    "created_at": r[3]
                } for r in rows
            ]

    def get_platform_stats(self) -> dict:
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM tenants")
            tenants = c.fetchone()[0]
            
            c.execute("SELECT COUNT(*) FROM projects")
            total_projects = c.fetchone()[0]
            
            c.execute("SELECT COUNT(*) FROM projects WHERE status = 'live'")
            live_projects = c.fetchone()[0]
            
            return {
                "tenants": tenants,
                "total_projects": total_projects,
                "live_projects": live_projects
            }
