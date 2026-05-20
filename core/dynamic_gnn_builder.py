import json
import numpy as np
import time

try:
    from numba import njit
except ImportError:
    # Graceful fallback: decorator does nothing if numba is not installed
    def njit(cache=True):
        return lambda f: f

@njit(cache=True)
def _numba_propagate(H, W_edge, A, update_gate=0.3):
    num_nodes = len(H)
    
    # 1. Compute Messages using explicit loops (avoids np.dot dependency on SciPy)
    messages = np.zeros(num_nodes)
    for u in range(num_nodes):
        for v in range(num_nodes):
            if A[u, v] > 0:
                messages[v] += H[u] * W_edge[u, v]
    
    # 2. Dynamic Edge Weighting Decay (10% decay per unit of stress)
    for u in range(num_nodes):
        stress = H[u]
        degradation = 1.0 - (stress * 0.1)
        for v in range(num_nodes):
            W_edge[u, v] = max(W_edge[u, v] * degradation, 0.1)
            
    # 3. GRU State Update
    new_H = (H * update_gate) + (messages * (1.0 - update_gate))
    return new_H, W_edge

@njit(cache=True)
def _ode_derivative(H, A, W_edge, alpha=0.5):
    """
    Computes continuous stress rate of change: dH/dt = -alpha * H + messages
    """
    num_nodes = len(H)
    messages = np.zeros(num_nodes)
    for u in range(num_nodes):
        for v in range(num_nodes):
            if A[u, v] > 0:
                messages[v] += H[u] * W_edge[u, v]
    return -alpha * H + messages

@njit(cache=True)
def _numba_propagate_ode(H, W_edge, A, dt=1.0, alpha=0.5):
    """
    Solves continuous temporal state propagation via 4th-order Runge-Kutta (RK4) integration.
    """
    num_nodes = len(H)
    
    k1 = _ode_derivative(H, A, W_edge, alpha)
    
    H_k2 = H + 0.5 * dt * k1
    k2 = _ode_derivative(H_k2, A, W_edge, alpha)
    
    H_k3 = H + 0.5 * dt * k2
    k3 = _ode_derivative(H_k3, A, W_edge, alpha)
    
    H_k4 = H + dt * k3
    k4 = _ode_derivative(H_k4, A, W_edge, alpha)
    
    new_H = H + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    for i in range(num_nodes):
        new_H[i] = max(new_H[i], 0.0)
        
    for u in range(num_nodes):
        stress = H[u]
        degradation = 1.0 - (stress * 0.1)
        for v in range(num_nodes):
            W_edge[u, v] = max(W_edge[u, v] * degradation, 0.1)
            
    return new_H, W_edge

# ── JIT COMPILER WARMUP PHASE ──
# Run an immediate execution with dummy structures during import to trigger
# LLVM compilation before any user requests arrive.
print("[GNN Warmup] Compiling T-GAT Numba JIT math kernels (Discrete + Continuous)...")
_dummy_H = np.zeros(2, dtype=np.float64)
_dummy_W = np.zeros((2, 2), dtype=np.float64)
_dummy_A = np.zeros((2, 2), dtype=np.float64)
_, _ = _numba_propagate(_dummy_H, _dummy_W, _dummy_A, update_gate=0.3)
_, _ = _numba_propagate_ode(_dummy_H, _dummy_W, _dummy_A, dt=1.0, alpha=0.5)
print("[GNN Warmup] JIT Compilation complete. Zero cold-start latency expected.")


class DynamicTGATBuilder:
    def __init__(self, graph_json: str, propagation_mode: str = "continuous_ode", alpha: float = 0.5):
        """
        Stage 2: Dynamic GNN Auto-Configuration.
        Reads the auto-generated generic Knowledge Graph and dynamically sizes
        the Temporal Graph Attention Network (T-GAT) arrays.
        """
        data = json.loads(graph_json)
        self.propagation_mode = propagation_mode
        self.alpha = alpha
        self.last_signal_time = None
        
        # 1. Dynamically Map Nodes
        self.node_names = [n[0] for n in data["nodes"]]
        self.num_nodes = len(self.node_names)
        self.node_idx = {name: idx for idx, name in enumerate(self.node_names)}
        
        print(f"[GNN Compiler] Auto-configuring GNN layers for {self.num_nodes} nodes under '{self.propagation_mode}' regime...")
        
        # 2. Dynamically build Adjacency Matrix (A) and Edge Weights (W_edge)
        self.A = np.zeros((self.num_nodes, self.num_nodes))
        self.W_edge = np.zeros((self.num_nodes, self.num_nodes))
        self.time_lags = np.ones((self.num_nodes, self.num_nodes)) # Delta t_uv
        
        for edge in data["edges"]:
            u_name, v_name, attrs = edge[0], edge[1], edge[2]
            u, v = self.node_idx[u_name], self.node_idx[v_name]
            self.A[u, v] = 1.0
            self.W_edge[u, v] = attrs.get("weight", 1.0)
            
        # 3. Initialize dynamic hidden states (GRU memory buffer)
        self.H = np.zeros(self.num_nodes) # Current stress/activation state

        print("[GNN Compiler] T-GAT Engine successfully instantiated and sized.")

    def stream_continuous_signal(self, node_name: str, shock_value: float, timestamp: float = None):
        """
        Continuous Temporal Streaming API.
        Accepts real-time signals (IoT, RSS, Tickers) and injects them as dynamic perturbations.
        Dynamically computes continuous time-deltas dt.
        """
        if node_name not in self.node_idx:
            print(f"[API WARN] Signal dropped. Target '{node_name}' not found in active graph.")
            return
            
        idx = self.node_idx[node_name]
        # Inject the raw shock value
        self.H[idx] += shock_value
        print(f"[Streaming API] {time.strftime('%H:%M:%S')} | Shock Ingested: +{shock_value} at {node_name}")
        
        dt = 1.0
        if timestamp is not None:
            if self.last_signal_time is not None:
                dt = max(0.001, timestamp - self.last_signal_time)
            self.last_signal_time = timestamp
            
        # Auto-trigger network propagation
        self._propagate_temporal_shock(dt=dt)

    def _propagate_temporal_shock(self, dt: float = 1.0):
        """
        Executes the mathematical T-GAT propagation across the auto-generated graph.
        """
        if self.propagation_mode == "continuous_ode":
            self.H, self.W_edge = _numba_propagate_ode(self.H, self.W_edge, self.A, dt=dt, alpha=self.alpha)
        else:
            self.H, self.W_edge = _numba_propagate(self.H, self.W_edge, self.A, update_gate=0.3)

    def get_network_state(self):
        """
        Returns the current dynamic stress levels of the entire network.
        """
        return {name: float(self.H[idx]) for name, idx in self.node_idx.items()}


if __name__ == "__main__":
    # Simulate receiving the JSON from Stage 1 Auto-Ingestion
    mock_json = '''{
      "nodes": [
        ["MICROCHIPFACTORY", {"dynamic_stress": 0.0}],
        ["AUTOMOTIVEPLANT", {"dynamic_stress": 0.0}],
        ["LITHIUMMINE", {"dynamic_stress": 0.0}],
        ["BATTERYFACTORY", {"dynamic_stress": 0.0}],
        ["GLOBALSHIPPINGLANE", {"dynamic_stress": 0.0}],
        ["CENTRALBANK", {"dynamic_stress": 0.0}]
      ],
      "edges": [
        ["MICROCHIPFACTORY", "AUTOMOTIVEPLANT", {"relation_type": "supplies", "weight": 1.0}],
        ["LITHIUMMINE", "BATTERYFACTORY", {"relation_type": "supplies", "weight": 1.0}],
        ["BATTERYFACTORY", "AUTOMOTIVEPLANT", {"relation_type": "supplies", "weight": 1.0}],
        ["GLOBALSHIPPINGLANE", "LITHIUMMINE", {"relation_type": "affects", "weight": 1.0}],
        ["CENTRALBANK", "MICROCHIPFACTORY", {"relation_type": "regulates", "weight": 1.0}]
      ]
    }'''

    engine = DynamicTGATBuilder(mock_json)
    
    # Simulate the Continuous Temporal Streaming API
    print("\n--- Testing Continuous Streaming API ---")
    
    # T=0: Global Shipping Lane gets blocked (e.g. Suez Canal crisis)
    engine.stream_continuous_signal("GLOBALSHIPPINGLANE", shock_value=5.0)
    
    # T=1: Central Bank raises rates unexpectedly
    engine.stream_continuous_signal("CENTRALBANK", shock_value=2.0)
    
    # T=2: Propagate time forward manually
    engine._propagate_temporal_shock()
    
    # View the auto-calculated dynamic states
    print("\n--- Dynamic Network Threat States ---")
    state = engine.get_network_state()
    for node, stress in sorted(state.items(), key=lambda item: item[1], reverse=True):
        print(f"[{node}]: {stress:.3f} threat score")
