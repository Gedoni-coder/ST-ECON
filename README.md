# Universal Neuro-Symbolic Intelligence Engine (IaaS)

A production-grade, highly optimized, multi-tenant Intelligence-as-a-Service (IaaS) platform. It ingests unstructured text, compiles it into spatial-temporal Knowledge Graphs, instantiates dynamic Graph Attention Networks (T-GAT), performs vectorized counterfactual simulations, and serves structured, LLM-generated advisory reports via a secure, durable, and thread-isolated REST API.

---

## Key Features

1. **Native Asynchronous NLP Ingestion (Stage 1)**: Parses raw text or unstructured documents into structured entity relationships and action spaces, natively integrated with FastAPI/Uvicorn asynchronous event loops.
2. **LLVM-Compiled Temporal GNN (Stage 2)**: Dynamically constructs and scales a Temporal Graph Attention Network (T-GAT) sized to the ingested topology, utilizing Numba JIT compiling to bypass Python execution bottlenecks.
3. **Closed-Loop Neuro-Symbolic Planning (Stage 3)**: Bridges reinforcement learning feedback directly back to planning by using live-learned Weak Supervision action weights to dynamically compute counterfactual threat reductions.
4. **Weak Supervision Learning Loop (Stage 4)**: Dynamically adjusts action weights based on real-world outcomes using Softmax-like probability scaling and L2 regularization to prevent weight divergence.
5. **TransFusion Hardware Acceleration (Stage 5)**: Implements custom compilers featuring DPipe scheduling, TileSeek tiling search under SRAM memory budgets, 1-pass attention bounds, and topological DAG liveness analyzers utilizing Belady's MIN optimal cache eviction algorithms.
6. **Thread-Isolated Local LLM execution (Stage 6)**: Runs local Hugging Face and LlamaCpp GGUF inference inside lazy class-level singleton ThreadPoolExecutors, protecting Uvicorn's main thread loop from GIL-starvation during generation.
7. **Durable Multi-Tenant Storage Layer**: Features a robust, transactional SQLite persistence manager that serializes network topologies, JIT states, and supervision logs to survive full server cold boots.

---

## Architectural Data Flow

```
 [ Raw KB Text ] ──► [ Stage 1: Async Ingest ] ──► [ Stage 2: Numba T-GAT GNN ]
                                                            │
                                                            ▼
 [ REST Client ] ◄── [ Stage 6: Isolated LLM ] ◄── [ Stage 3: Closed-Loop Planner ]
                                                            ▲
                                                            │
 [ Feedback Webhook ] ─────────────────────────────► [ Stage 4: Weak Supervision ]
```

---

## Installation & Setup

### 1. Install Dependencies
Make sure you have Git and Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Run Standalone Pipeline Demo
Execute the full neuro-symbolic pipeline from ingestion to prompt generation:
```bash
python run_pipeline.py
```

### 3. Run Durability and Integration Tests
Validate database reload durability and runtime stability:
```bash
python test_persistence.py
python final_test.py
```

### 4. Start the Hosting Engine
Run the FastAPI gateway on your preferred port:
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```
