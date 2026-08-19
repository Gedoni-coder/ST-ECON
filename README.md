# ST-ECON

A prototype pipeline that ingests unstructured economic text, builds a temporal
graph over the entities it finds, propagates state across that graph, and uses a
local language model to turn the result into structured advisory output. Served
behind a FastAPI gateway with per-tenant project isolation.

**This is a learning project.** It was built to understand the mechanics of
temporal graph networks, efficient attention, and neuro-symbolic pipelines by
implementing them rather than reading about them. It has not been validated
against real data, benchmarked against baselines, or deployed. Treat every
component as an exercise unless this file says otherwise.

## What actually runs

| Component | File | Status |
|---|---|---|
| Text → entity/relation ingestion | `core/auto_ingestion.py` | works, unvalidated quality |
| Temporal graph attention network | `core/dynamic_gnn_builder.py` | works, numba-accelerated |
| Symbolic verification layer | `core/symbolic_verifier.py` | works |
| Weak-supervision feedback loop | `core/feedback_loop.py` | works, no ground truth to learn from |
| Local LLM inference | `core/llm_client.py` | works (HF / llama.cpp GGUF) |
| Hardware acceleration modelling | `core/transfusion_accelerator.py` | mixed — see corrections below |
| Multi-tenant registry | `core/tenant_registry.py` | works, SQLite-backed |
| REST gateway | `server.py` | works |

`core/dynamic_gnn_builder.py` compiles its propagation kernels with numba
(`@njit`), which lowers through LLVM, and falls back to pure Python when numba
is unavailable. An earlier version of this file described that as an
"LLVM-compiled temporal GNN," which is technically true but overstated: it is
JIT-compiled numeric Python, not a custom compiler.

## Edge-inference experiments

Standalone implementations in `experiments/`, written to understand the
techniques. See [`experiments/README.md`](experiments/README.md) for details and
for corrections to earlier claims.

**`block_tiled_attention.py`** — tiled multi-head attention with online-softmax
rescaling, FlashAttention-style. Peak activation memory is bounded by block size
rather than growing with `seq_len²`. Verified against reference attention to
~1e-16 across sequence lengths, block sizes, ragged tails, and large-magnitude
inputs; `experiments/tests/` covers all of these.

This file was previously named `mas_attention.py` and its first version was
numerically wrong — it took the softmax maximum per key block while sharing
accumulators across blocks, giving errors of 0.26–0.54 against reference
attention. The old test used `seq_len=64` with a block size that resolved to a
single block, where the bug cannot appear, so it passed. The test suite now
reconstructs the broken formulation and asserts that it fails.

It is also not MAS-Attention. MAS-Attention (Shakerdargah, Lu, Gao & Niu, MLSys
2025) schedules attention across heterogeneous compute units; this is
single-device tiling.

**`quantize_model.py`** — PyTorch post-training static quantization and
quantization-aware training, using `QuantStub`/`DeQuantStub` with the fbgemm
qconfig. FP32→INT8 is a 4× reduction in weight storage by construction; **no
size or latency measurement has been taken here**, and no accuracy comparison
has been run.

**`attention_tiling.cu`** — CUDA tiled matrix multiplication using thread-block
shared memory with 16×16 tiles. Tiling reduces global memory traffic by roughly a
factor of `TILE_WIDTH`, i.e. from O(M·N·K) to O(M·N·K / TILE_WIDTH). An earlier
version of this file claimed a reduction to O(M·N), which is wrong — that would
require reading each input exactly once, which tiled matmul does not do. **The
kernel has not been benchmarked against cuBLAS.**

## Corrections to earlier claims

Recorded rather than deleted, because the audit that produced them is more useful
than a clean-looking README.

- **TileSeek is random search, not MCTS.** `search()` samples tile shapes
  uniformly from a candidate list under a buffer constraint. There is no tree,
  no visit counts, no selection policy, no backpropagation. It also does not
  converge: five consecutive runs at identical settings returned `tile_p` of 1,
  32, 128, 128, 64.
- **DPipe is greedy list scheduling, not dynamic programming.** One forward pass
  assigning nodes to two resource timelines by PE type. No DP table, no
  recurrence.
- **DPipe currently yields no speedup, and that is the interesting part.** At the
  default configuration it saves 0.002 µs out of 31.3 µs — a speedup of 1.00.
  Its own output explains why: the 2D matmul units carry 98.1% of total latency
  and the 1D vector units 1.9%. Two-resource pipelining only pays when the
  subgraphs carry comparable load. Here they do not.
- **Cache eviction is furthest-last-use, not Belady's MIN.** Belady evicts by
  furthest *next* use; this ranks by each tensor's *last* reader, which differs
  whenever a tensor has several readers. A reasonable approximation, not the
  optimal policy.
- **`OnePassAttention` in `transfusion_accelerator.py` is correct** — running max
  with rescaling, agreeing with unfused attention to ~5e-16 at every tile size,
  and it self-checks on each call.
- **Paper attribution.** TransFusion (MICRO 2025) and MAS-Attention (MLSys 2025)
  are separate papers. The mechanisms modelled in `transfusion_accelerator.py`
  follow TransFusion; an earlier docstring conflated the two.

## Known limitations

- No evaluation against real economic data. The pipeline has never been scored
  on a forecasting or advisory task with ground truth.
- The weak-supervision loop adjusts action weights from feedback, but no labelled
  outcomes exist to supply that feedback.
- The accelerator module is an analytical model of a hardware schedule. It
  estimates latencies from operand shapes and a clock parameter; it does not
  measure anything on real silicon.
- "Multi-tenant" means per-organisation API keys and project isolation in SQLite.
  It has not been load-tested or security-reviewed.

## Setup

```bash
pip install -r requirements.txt

python run_pipeline.py                  # ingestion → graph → advisory output
python -m uvicorn server:app --port 8000  # REST gateway

cd experiments && pytest tests/ -v      # attention correctness suite
```
