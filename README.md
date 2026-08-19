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

## Repository map

The Status column distinguishes what has been tested from what has only been
written. "Tested" means there is a check in this repository that would fail if
the component broke.

| Component | File | Status |
|---|---|---|
| Tiled attention + test suite | `experiments/block_tiled_attention.py` | **tested** — see `experiments/tests/` |
| Hardware acceleration modelling | `core/transfusion_accelerator.py` | **partly tested** — see corrections below |
| Temporal graph attention network | `core/dynamic_gnn_builder.py` | runs; numba-accelerated, no correctness test |
| Text → entity/relation ingestion | `core/auto_ingestion.py` | written, output quality unevaluated |
| Symbolic verification layer | `core/symbolic_verifier.py` | written, no test |
| Weak-supervision feedback loop | `core/feedback_loop.py` | written; no labelled outcomes exist to learn from |
| Local LLM inference | `core/llm_client.py` | written; Hugging Face and llama.cpp GGUF backends |
| Multi-tenant registry | `core/tenant_registry.py` | written, SQLite-backed |
| Orchestration | `core/orchestrator.py` | written |
| REST gateway | `server.py` | written |
| Browser dashboard | `ui/` | static JS/HTML/CSS client for the gateway |
| Integration / durability scripts | `final_test.py`, `test_persistence.py`, `test_platform.py`, `test_verification.py` | scripts, not a pytest suite |

`core/dynamic_gnn_builder.py` compiles its propagation kernels with numba
(`@njit`), which lowers through LLVM, and falls back to pure Python when numba is
unavailable. An earlier version of this README called that an "LLVM-compiled
temporal GNN" — technically true, but overstated: it is JIT-compiled numeric
Python, not a custom compiler.

## Edge-inference experiments

Standalone implementations in `experiments/`, written to understand the
techniques. See [`experiments/README.md`](experiments/README.md) for detail and
for corrections to earlier claims.

**`block_tiled_attention.py`** — tiled multi-head attention with online-softmax
rescaling, FlashAttention-style. Peak activation memory is bounded by block size
rather than growing with `seq_len²`. It agrees with unfused reference attention
to within float32 precision across sequence lengths, block sizes, ragged tails,
gradient flow, and inputs scaled to force overflow; `experiments/tests/` asserts
all of these against a 1e-4 tolerance.

This file was previously named `mas_attention.py`, and its first version was
numerically wrong: it took the softmax maximum per key block while sharing
accumulators across blocks, so terms computed under different maxima were summed
together. Measured against reference attention, outputs were off by 0.26–0.54 in
absolute value. It went unnoticed because the only test used `seq_len=64` with a
block size that resolved to a single block — the one configuration where the bug
cannot appear. The suite now reconstructs the broken formulation and asserts that
it *fails*, so the regression cannot return quietly.

It is also not MAS-Attention. MAS-Attention (Shakerdargah, Lu, Gao & Niu, MLSys
2025) schedules attention across heterogeneous compute units; this is
single-device tiling.

**`quantize_model.py`** — PyTorch post-training static quantization and
quantization-aware training, using `QuantStub`/`DeQuantStub` with the fbgemm
qconfig. FP32→INT8 is a 4× reduction in weight storage by construction. **No size
or latency measurement has been taken, and no accuracy comparison has been run.**

**`attention_tiling.cu`** — CUDA tiled matrix multiplication using thread-block
shared memory with 16×16 tiles. Tiling reduces global memory traffic by roughly a
factor of `TILE_WIDTH` — from O(M·N·K) to O(M·N·K / TILE_WIDTH). An earlier
version of this README claimed a reduction to O(M·N), which is wrong; that would
require reading each input exactly once, which tiled matmul does not do. **The
kernel has not been benchmarked against cuBLAS.**

## Corrections to earlier claims

Recorded rather than deleted, because the audit that produced them is more useful
than a clean-looking README.

- **TileSeek is random search, not MCTS.** `search()` samples tile shapes
  uniformly from a candidate list under a buffer constraint. There is no tree, no
  visit counts, no selection policy, no backpropagation. It is also unseeded, so
  repeated runs at identical settings return different tile shapes — it does not
  converge on an answer.
- **DPipe is greedy list scheduling, not dynamic programming.** One forward pass
  assigning nodes to two resource timelines by PE type. No DP table, no
  recurrence.
- **DPipe currently yields no speedup, and that is the interesting part.** At the
  default configuration it saves 0.002 µs out of 31.3 µs — a speedup of 1.00. Its
  own output explains why: the 2D matmul units carry 98.1% of total latency and
  the 1D vector units 1.9%. Two-resource pipelining only pays when the subgraphs
  carry comparable load. Here they do not.
- **Cache eviction is furthest-last-use, not Belady's MIN.** Belady evicts by
  furthest *next* use; this ranks by each tensor's *last* reader, which differs
  whenever a tensor has several readers. A reasonable approximation, not the
  optimal policy. It does behave sensibly at the boundaries — 61% on-chip
  retention at a 64 KB budget, 0% at 1 byte.
- **`OnePassAttention` in `transfusion_accelerator.py` is correct.** Running max
  with rescaling; it self-checks against unfused attention on every call and
  reports the deviation, which stays at the floating-point noise floor for every
  tile size tested.
- **Paper attribution.** TransFusion (MICRO 2025) and MAS-Attention (MLSys 2025)
  are separate papers. The mechanisms modelled in `transfusion_accelerator.py`
  follow TransFusion; an earlier docstring conflated the two.

## Known limitations

- No evaluation against real economic data. The pipeline has never been scored on
  a forecasting or advisory task with ground truth.
- The weak-supervision loop adjusts action weights from feedback, but no labelled
  outcomes exist to supply that feedback.
- The accelerator module is an analytical model of a hardware schedule. It
  estimates latencies from operand shapes and a clock parameter; it measures
  nothing on real silicon.
- "Multi-tenant" means per-organisation API keys and project isolation in SQLite.
  Not load-tested, not security-reviewed.
- Outside `experiments/`, the `test_*.py` files are runnable scripts rather than
  an assertion-based suite, so a silent regression in `core/` would not be caught.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt

python run_pipeline.py                    # ingestion → graph → advisory output
python -m uvicorn server:app --port 8000  # REST gateway, dashboard in ui/

python final_test.py                      # integration script
python test_persistence.py                # SQLite durability across restarts

cd experiments
pip install -r requirements.txt
pytest tests/ -v                          # attention correctness suite
```

`numba` is optional — `dynamic_gnn_builder` falls back to pure Python without it,
more slowly. A local LLM backend (Hugging Face `transformers` or `llama-cpp-python`)
is needed only for the advisory generation stage; `OllamaModelfile.template` is a
starting point for serving a model through Ollama instead.
